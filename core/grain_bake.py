"""Tables the GPU grain stage needs, baked from the engine's own pipeline.

Two things live here:

* **The density -> RGB cube.**  The grain law is a function of *negative density*, and the
  density has to be perturbed before the print runs.  A single RGB->RGB LUT cannot express
  that (it never exposes the density), so the chain is split in two:

      linear RGB --[density cube]--> C/M/Y density --+grain--> --[print cube]--> RGB out

  The first cube already exists (``SpektraEngine._bake_film_density``); this module bakes the
  second one by injecting a uniform C/M/Y lattice at the ``cmy_film`` tap, which is where the
  reference inserts grain too.

* **The sub-layer table.**  ``spektrafilm.model.density_curves.interp_density_cmy_layers``
  splits a pixel's total density into the three coated sub-layers by interpolating the
  profile's ``density_curves_layers`` against ``density_curves``.  On the GPU that is a texture
  fetch, so the interpolation is resampled here onto a *uniform* density axis -- no search in
  the shader, just a multiply.

Nothing here touches the kernel: the cubes are produced by running its pipeline, and the
tables come out of the profile JSON the same way the reference reads them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalized_density_curves",
    "density_axis_range",
    "bake_density_to_rgb",
    "bake_layer_table",
    "layer_scalars",
    "grain_kernel_1d",
]

# The shader clamps its density lookup to the axis this grid covers, so the grid has to be a
# little wider than the densities the image actually reaches: grain pushes density either way.
_DENSITY_MARGIN = 0.15
_MIN_SPAN = 0.2


def normalized_density_curves(film_data):
    """The film's curves as ``develop()`` uses them: baseline-zeroed.

    ``spektrafilm.model.develop.develop`` subtracts each record's minimum before interpolating,
    and ``interp_density_cmy_layers`` is fed that same normalized curve, so the grain axis and
    the density the LUT carries are on one footing only if we do it here too.
    """
    curves = np.asarray(film_data.density_curves, dtype=np.float64)
    return curves - np.nanmin(curves, axis=0)


def density_axis_range(density_cube, film_data=None):
    """Per-channel ``(min, max)`` for the print cube's lattice.

    Taken from the densities the bake actually produced rather than from the curve endpoints:
    a stock never reaches its own Dmax across the whole LUT lattice, and a tight lattice is
    what keeps 33 nodes enough.  The result is widened by a fixed margin so grain has somewhere
    to go, and clipped at zero -- density is not negative.
    """
    cube = np.asarray(density_cube)
    lo = np.nanmin(cube.reshape(-1, 3), axis=0).astype(np.float64) - _DENSITY_MARGIN
    hi = np.nanmax(cube.reshape(-1, 3), axis=0).astype(np.float64) + 0.05
    lo = np.maximum(lo, 0.0)
    # A channel that barely moves would give the cube a lattice thinner than the grain it has
    # to carry; give it a floor so the interpolation stays sane.
    span = np.maximum(hi - lo, _MIN_SPAN)
    hi = lo + span
    if film_data is not None:
        curve_max = np.nanmax(normalized_density_curves(film_data), axis=0)
        hi = np.minimum(hi, np.maximum(curve_max + _DENSITY_MARGIN, lo + _MIN_SPAN))
    return lo.astype(np.float32), hi.astype(np.float32)


def bake_density_to_rgb(sim, lut_size, dens_min, dens_max):
    """Bake ``C/M/Y density -> output RGB`` by injecting a uniform lattice at ``cmy_film``.

    The lattice is built as a 2-D image ``(N, N*N, 3)`` because that is how the rest of the
    engine feeds the pipeline (it is what the numba stages parallelise over), and the flat
    order matches C-order of the ``(N, N, N, 3)`` cube we return.

    Axis order: ``cube[Y, M, C]`` -- numpy's last axis is a 3-D texture's fastest (x) axis,
    so this is the array shape that makes ``texture(cube, vec3(C, M, Y))`` sample the right
    point.  Channel order inside a texel stays C, M, Y, exactly as ``develop`` returns them.
    """
    from spektrafilm.runtime.topology import Tap

    n = int(lut_size)
    c_axis = np.linspace(float(dens_min[0]), float(dens_max[0]), n, dtype=np.float32)
    m_axis = np.linspace(float(dens_min[1]), float(dens_max[1]), n, dtype=np.float32)
    y_axis = np.linspace(float(dens_min[2]), float(dens_max[2]), n, dtype=np.float32)

    # meshgrid order sets the array's axis order, so ask for [Y, M, C].
    yg, mg, cg = np.meshgrid(y_axis, m_axis, c_axis, indexing="ij")
    lattice = np.stack([cg, mg, yg], axis=-1).astype(np.float32)

    flat = lattice.reshape(n, n * n, 3)
    out = sim._pipeline.process(flat, inject=Tap.CMY_FILM, collect=Tap.RGB_OUT)
    cube = np.ascontiguousarray(np.asarray(out, dtype=np.float32).reshape(n, n, n, 3))
    return np.clip(cube, 0.0, 1.0)


def bake_layer_table(film_data, samples=256):
    """Resample the sub-layer split onto a uniform density axis, ready for a 3-D texture.

    Returns ``dict`` with

    ``axis_min`` / ``axis_max``  (3,)  -- the density range each record is resampled over
    ``table``                    (samples, 3, 4) -- (layer0, layer1, layer2, 0) per record

    Column order inside ``table`` matches ``density_curves_layers[:, sl, ch]``: the three
    coated layers of record ``ch``, fast -> slow.
    """
    axis = normalized_density_curves(film_data)                     # (M, 3)
    layers = np.asarray(film_data.density_curves_layers, dtype=np.float64)  # (M, 3, 3)

    lo = axis.min(axis=0)
    hi = axis.max(axis=0)
    uniform = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    table = np.zeros((samples, 3, 4), dtype=np.float32)

    for ch in range(3):
        x = axis[:, ch]
        # np.interp needs an increasing axis; a stock with a flat record would break it.
        order = np.argsort(x)
        xs = x[order]
        for sl in range(3):
            ys = layers[order, sl, ch]
            values = np.interp(uniform * (hi[ch] - lo[ch]) + lo[ch], xs, ys)
            table[:, ch, sl] = values.astype(np.float32)

    return {
        "axis_min": lo.astype(np.float32),
        "axis_max": hi.astype(np.float32),
        "table": table,
    }


def grain_kernel_1d(pixel_size_um, final_blur_px=0.65, radius=2, decay_um=1.45, unsharp=None):
    """The discrete kernel the reference's dye-cloud cascade reduces to at this pitch.

    ``apply_grain_to_density_layers`` ends with two convolutions: the four Jarvis dye-cloud
    gaussians (weights ``JARVIS_DYE_CLOUD_GAUSSIANS``, sigmas scaled by ``decay_um/pitch``),
    then the frame blur ``grain.blur``.  Both are isotropic, so together they are one
    separable kernel -- but the *discrete* weights are not the sampled gaussian: scipy's
    ``gaussian_filter`` truncates and renormalises each of the four terms at its own radius.

    ``unsharp`` folds the scanner's sharpening into the same kernel.  ``scan()`` runs
    ``unsharp_mask(sigma, amount)`` on the display RGB *after* the grain has been developed,
    and a pointwise print cube cannot carry a spatial filter at all -- so left out, the frame
    loses 34 % of its high-frequency energy, grain included (measured: the reference's whole
    frame comes out at sigma 8.77/9.94/10.27 through the scan, 6.58/7.35/7.58 with the mask
    off).  Folding it here is exact for the grain: the density -> RGB cube is pointwise, so a
    linear operator applied after it equals the same operator applied to the density
    perturbation with the local slope as a scalar.

    Rather than re-derive any of that (the first attempt at doing so came out 13 % too wide
    and cost 0.87x the reference's sigma), the cascade is simply run on a delta here and the
    centre row read back.  The numbers handed to the shader are therefore the reference's
    own, at the resolution being rendered.

    Returns ``radius*2+1`` weights that sum to 1; the 2-D kernel is their outer product.
    """
    from scipy.ndimage import gaussian_filter

    pad = 8
    n = 2 * (radius + pad) + 1
    delta = np.zeros((n, n), dtype=np.float64)
    delta[radius + pad, radius + pad] = 1.0

    decay_px = float(decay_um) / max(float(pixel_size_um), 0.1)
    out = np.zeros_like(delta)
    for weight, factor in ((0.240, 0.518), (0.384, 0.983), (0.274, 1.848), (0.102, 3.731)):
        out += weight * gaussian_filter(delta, max(decay_px * factor, 0.1))
    if final_blur_px and final_blur_px > 0:
        out = gaussian_filter(out, float(final_blur_px))
    if unsharp is not None:
        sigma, amount = float(unsharp[0]), float(unsharp[1])
        if sigma > 0 and amount > 0:
            from spektrafilm.model.diffusion import apply_unsharp_mask
            out = apply_unsharp_mask(np.repeat(out[..., None], 3, axis=2), sigma, amount)[..., 0]

    centre = radius + pad
    row = out[centre, centre - radius:centre + radius + 1].astype(np.float64)
    total = row.sum()
    if total <= 0:
        row = np.zeros(2 * radius + 1); row[radius] = 1.0
        return row.astype(np.float32)
    return (row / total).astype(np.float32)


def layer_scalars(film_data, grain, pixel_size_um):
    """Per (record, sub-layer) numbers the shader needs besides the table.

    This is ``spektrafilm.model.grain.apply_grain_to_density_layers`` transcribed, minus the
    sampling itself:

      * ``density_max_layers[sl, ch]`` -- the layer's ceiling, *plus* its share of the fog;
      * ``n_particles[sl, ch]``        -- particles per pixel, split by layer area;
      * ``blur_sigma[sl, ch]``         -- dye-cloud blur, in pixels, from ``sqrt(od_particle)``.

    A last, frame-wide blur (``grain.blur``) is applied to the summed result and is reported
    separately as ``final_blur_px``.
    """
    layers = np.asarray(film_data.density_curves_layers, dtype=np.float64)
    density_max_layers = np.nanmax(layers, axis=0)                      # (3 sl, 3 ch)
    total = density_max_layers.sum(axis=0)
    fractions = density_max_layers / np.maximum(total, 1e-9)            # (3, 3)

    density_min = np.asarray(grain.density_min, dtype=np.float64)       # (3,)
    density_min_layers = fractions * density_min[None, :]
    density_max_abs = density_max_layers + density_min_layers

    if float(grain.particle_area_um2) <= 0.0:
        # The bake path zeroes this when grain is off (`digest_params`), so a caller that
        # reached here with a digested parameter set would silently divide to infinity.
        raise ValueError("grain.particle_area_um2 is 0 -- this is a grain-disabled parameter set")
    pixel_area = float(pixel_size_um) ** 2
    particle_area = (float(grain.particle_area_um2)
                     * np.asarray(grain.particle_scale, dtype=np.float64)[None, :]
                     * np.asarray(grain.particle_scale_layers, dtype=np.float64)[:, None])
    particles = pixel_area * fractions / particle_area                  # (3 sl, 3 ch)

    od_particle = density_max_abs / np.maximum(particles, 1e-9)
    blur_sigma = float(grain.blur_dye_clouds_um) * np.sqrt(np.maximum(od_particle, 0.0))

    return {
        "density_min": density_min.astype(np.float32),
        "density_min_layers": density_min_layers.astype(np.float32),
        "density_max_layers": density_max_abs.astype(np.float32),
        "particles": particles.astype(np.float32),
        # Particles scale with the *square* of the pixel pitch, so the shader needs the per-um^2
        # rate rather than a number frozen at one render size: an export at 46 MP and a preview
        # at 1 MP are different samplings of the same film.
        "particles_per_um2": (fractions / particle_area).astype(np.float32),
        "uniformity": np.asarray(grain.uniformity, dtype=np.float32),
        # How far the three records keep their own dye clouds (`mix_chroma_grain` upstream).
        # 0.5 is the reference: half independent, half shared luminance.
        "colour_grain": float(getattr(grain, "colour_grain", 0.5)),
        "blur_sigma": blur_sigma.astype(np.float32),
        "blur_um": float(grain.blur_dye_clouds_um),
        "final_blur_px": float(grain.blur),
    }
