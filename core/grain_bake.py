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

# Frame-level blur applied after the dye-cloud cascade, in *pixels*.  The reference uses
# ``GrainParams.blur = 0.65`` to hide the aliasing of per-pixel grain sampling, but that is
# an anti-aliasing approximation, not physics: on fotufilm's own step chart (same 12 um pitch,
# gold 200 + ektacolor edge) 0.65 puts the grain one full step wider than the reference.
#
# Its job used to be to carry the whole correlation, because the reference's 1.45 um dye cloud
# is 0.25 px at export pitch and does nothing there.  Now that the clump is modelled properly
# (see ``DYE_CLOUD_DECAY_UM``) this is only the residual: at 5.81 um/px the cascade alone sits
# at lag-1 0.60 and 0.45 px of frame blur takes it to 0.685, against the reference's 0.706.
#
# Because it is expressed in pixels rather than micrometres it does not scale with pitch, so
# at a 60 um/px preview it is the *only* term left and the grain is again 1 px white noise
# (0.172) -- which is what a 60 um sampling of a 6 um clump can resolve.
FRAME_BLUR_PX = 0.45

# ---------------------------------------------------------------------------------------
# Grain scale, as fotufilm draws gold 200 at 1:1 (the 4024 x 6024 export is 5.81 um/px
# across the short edge -- measured on the probe, `_preview_export_parity.py`).
#
# The reference's dye-cloud decay is a single 1.45 um for all 61 stocks -- a format default,
# not a per-film calibration -- and at 5.81 um/px that puts the noise at a lag-1
# autocorrelation of 0.24 with a 1 px half width.  fotufilm's own render of the same frame
# sits at 0.706 / 2 px: its grains are *clumps*, several pixels across, not per-pixel white
# noise.
#
# ---------------------------------------------------------------------------------------
# Grain scale, anchored on gold 200 against fotufilm on the neutral test chart
# (`测试/中性测试图_6000x4000.png`, 6.0 um/px on the reference, 5.83 on our 35 mm long edge).
# The chart supersedes the earlier live-frame readings: 24 blocks, both engines paired on
# output L*, so curve slope and subject colour no longer leak into the numbers.  Measured
# there on the midtone grey blocks (`scratch/_chart_compare.py`):
#
#     fotufilm   sigma R/G/B 0.0174/0.0238/0.0349   lag-1 0.671/0.664/0.658   half 2 px
#     ours (old) sigma R/G/B 0.0225/0.0146/0.0228   lag-1 0.594/0.593/0.598   half 2 px
#
# ``DYE_CLOUD_DECAY_UM`` sets the correlation width.  On the shader's own kernel at our
# 5.83 um/px pitch (with the scanner unsharp folded in, `grain_kernel_1d`): 6.0 -> 0.599,
# 6.6 -> 0.657, 6.7 -> 0.665, 6.8 -> 0.673.  6.7 um lands on the chart's 0.664.  (The
# 0.706 quoted in earlier rounds was the live-frame residual on a blue sky -- same physics,
# different window; the chart is the controlled read and the anchor from here on.)
#
# ``CHANNEL_RATE_SCALE`` fixes the per-channel balance.  fotufilm's B record is the loudest
# on every block of the chart (0.50 : 0.68 : 1 normalised to B); ours came out 0.98 : 0.64 : 1
# because the particles live in CMY density and the print cube's per-channel slope reshapes
# them on the way out -- so the archive's ``layer_weights`` cannot be mapped linearly and the
# balance is a measured multiplier on the particle *rate*.  Sigma goes as ONE OVER the square
# root of the rate (more, smaller particles = less RMS, Selwyn), which the first iteration
# confirmed exactly: rate x3.24 on M moved the output G sigma by 0.552 = 1/sqrt(3.24).  The
# values below are the first-order solve at the chart's midtone (D1), including the ~0.95
# sigma cost of widening the kernel 6.0 -> 6.7 um; the chart re-measurement is the judge.
#
# Both are gold-200 ANCHOR values carried as globals: the only stock calibrated so far.
# The per-stock resolver (field wiring) replaces their *input* -- constants -> archive
# fields x these coefficients -- with this chart result as the regression target; the
# downstream bundle/shader path is parameterised already and does not change.
DYE_CLOUD_DECAY_UM = 6.7
CHANNEL_RATE_SCALE = (1.19, 0.36, 0.47)
COLOUR_GRAIN = 1.0

# ---------------------------------------------------------------------------------------
# Monochrome stocks take their own path -- fotufilm's B&W renders measure identical grain
# on TriX and HP5 (lag-1 0.556 / 0.561, equal output sigma 0.032, channel corr +0.99)
# regardless of their nominal size/strength fields, so the archive fields are ignored here
# and three measured constants stand in.
#
#   * ``BW_DYE_CLOUD_DECAY_UM`` -- kernel lag-1 at 5.83 um/px with the scanner unsharp
#     folded in: 5.2 -> 0.509, 5.6 -> 0.556, 5.8 -> 0.578.  5.6 lands between the two
#     measured references; 8 um of *size_mm* would have predicted 10.7 um and a lag of 0.80,
#     i.e. the size mapping does not extend to the mono path.
#   * ``BW_RATE_SCALE`` -- per-record rate over the reference's own particle scale.  The
#     colour mapping would hand weights (1,1,1) the rate (1.7, 0.36, 0.35), which printed
#     the output R channel nearly silent (sigma 0.007 against G 0.058) -- colour speckle
#     on a black-and-white print.  Equal records plus the shared draw fix the structure;
#     the overall scale is set against fotufilm's midtone sigma of 0.032.
BW_DYE_CLOUD_DECAY_UM = 5.6
# Overall amplitude: measured on the chart at matched OUTPUT L* (the tone curves differ
# between the engines, so sigma must be compared at the same print level, not the same
# block) ours ran 1.27-1.40x hot across the whole scale; sigma goes as 1/sqrt(rate), and
# 1.34^2 = 1.8.  The per-record rates are deliberately equal: the probe renders showed
# each record's rate moves all three output channels alike, so rates cannot -- and need
# not -- decouple the residual few-percent channel imbalance that the colour paper's own
# slope mismatch produces.
#
# 2026-11-07: the kernel constant above stands, but the **amplitude** no longer ignores the
# archive.  The "identical grain on TriX and HP5" evidence was two 400-speed stocks whose
# fields are nearly equal (0.017 / 0.016) -- generalising that to the whole family is what
# printed Delta 3200 (strength 0.019, the highest of the seven) as the *quietest* of them.
# Measured against the engine on the correct paper (vision-2383, §41), our sigma tracked the
# print's density ceiling (corr 0.85) while the engine tracks the emulsion's strength
# (corr 0.84).  So `area_mul` now carries the strength and `_read_density_rate` takes the
# depth of the print back out -- the same two things the colour branch does.
# 2026-11-07 重新标定：上面两项（strength + _read_density_rate）接进来之后，七支卷实测
# 比值收到 0.29–0.39（原来 0.34–1.25），几何均值 0.35；sigma ∝ rate^-0.5，故
# 1.8 × 0.35² = 0.22。旧值 1.8 是在"幅度跟着 Dmax 走"的那套模型下标出来的，不再适用。
BW_RATE_SCALE = (0.22, 0.22, 0.22)
# The strength the rate above was calibrated against: the mean of the TriX / HP5 pair the
# mono path was originally measured on, so those two keep their old amplitude.
BW_ANCHOR_STRENGTH = 0.0165

# ---------------------------------------------------------------------------------------
# Group-wide grain *width*, as a factor on the dye-cloud cascade (folded into the shader's
# `grain_size` slider, whose implementation compensates the rate so sigma is invariant).
#
# Measured 2026-11-07 on real frames, shape = sigma@16px / sigma@2px:
#
#   group            ours    engine   ratio
#   colour negative  1.00    1.05     0.93-1.06
#   reversal         1.18    1.05     1.09-1.13   (flat across the whole tone scale)
#   monochrome       1.18    1.04     1.11-1.15
#
# So the two groups that carry their own rate/size calibration render ~11% coarser, while
# the colour negatives -- which never had that patchwork -- are already right.  The archive's
# `size_mm` is NOT the lever: scaling it 15% down moved shape only 0.03 and pushed sigma up
# 18%.  `FRAME_BLUR_PX` is not either: going 0.45 -> 0.65 (the upstream value) dropped sigma
# 30% and made the shape *worse* (1.07 -> 1.18).  The slider works because it resamples the
# whole cascade rather than one term of it.
#
# 0.65 was measured: shape 1.07 -> 1.00 (velvia), 1.10 -> 1.02 (tri_x), sigma within 5-9%.
GROUP_KERNEL_SCALE = 0.65


def _group_kernel_scale(is_bw, is_reversal):
    return GROUP_KERNEL_SCALE if (is_bw or is_reversal) else 1.0


# The stock whose measured chart calibration fixes the three constants above.  When a stock
# carries fotufilm crystal-grain archive fields (`physical_extensions.crystal_grain`), the
# resolver below scales the anchor by the field ratios -- linear first-order, refined later
# by per-stock measurement.  The anchor stock itself divides by its own fields, i.e. is
# bit-identical to the constants, which is the regression that guards the wiring.
# fotufilm's layer_weights run (red-, green-, blue-sensitive) and its docs call the
# blue-sensitive layer the grainiest -- index 2 of gold 200's [0.7, 1, 1.35] -- which maps
# onto our density channels (C, M, Y) one to one: the red-sensitive record lays cyan.
CRYSTAL_ANCHOR = {"size_mm": 0.005, "strength": 0.0105, "weights": (0.7, 1.0, 1.35)}

# The anchor stock's per-channel total layer density -- the sum of
# ``density_curves_layers`` along the sub-layer axis -- for gold 200 behind the reference
# process.  It exists so the amplitude can be anchored on ``strength`` alone.
#
# Why it is needed: the sampled sigma works out to ``dMax * sqrt(sat*p/n_particles)``, and
# ``n_particles`` carries no density term, so the port read ``sigma ~ dMax * strength``.
# fotufilm does not: ``readSigma = grainStrength * grainLayerWeights[l]`` and
# ``anchorToSheet`` iterates the rendered sigma onto it at the *read density*, so how deep a
# stock can go never enters the amplitude.  Dividing the rate by ``(total/total_anchor)^2``
# cancels the dMax factor (``sigma ~ dMax/sqrt(rate)``, and ``dMax = fraction*total``).
#
# Measured 2026-11-07 on three stocks whose size/strength/weights/law are all identical --
# pro_400h, c200, superia_200 -- and whose totals are 1.61/1.64/2.15, 1.73/2.07/2.67 and
# 2.93/2.93/2.93: before this they came out 0.80 / 0.87 / 1.38 apart, which is a ratio of
# 1.7 between stocks fotufilm renders identically.  See docs §26.3.
CRYSTAL_ANCHOR_TOTAL = (1.6269010800000001, 1.5136206699999999, 1.6861682499999999)

# -- granularity against density -----------------------------------------------------------
#
# fotufilm's published RMS figure is read at one density; away from it the emulsion's own law
# decides how loud the grain is (`FilmStock.grainDensityModulation`).  A colour negative is
# not monotone -- the measured shape rises to a peak 0.15-0.2 above D-min, falls, and rises
# again near net 1.1-1.5 where the slow sub-layer comes in -- so this cannot be folded into
# the strength constant.  Both renderers evaluate the same closed form per pixel.
#
# The three numbers below are fotufilm's own defaults (`FilmStock.defaultGrainFogDensity`,
# `defaultGrainReversalProfile`, `FilmStockDefaults.grainDensityRecords`); a profile overrides
# only the profile, the law and the anchor.
GRAIN_FOG_DENSITY = 0.03
GRAIN_REVERSAL_PROFILE = (1.1, 3.0)
GRAIN_DENSITY_PROFILE_DEFAULT = (
    (20.0, 0.3373, 0.2629, 0.8187, 1.1811, 0.3212),
    (20.0, 0.3583, 0.3183, 1.6514, 1.4856, 0.3631),
    (20.0, 0.1069, 0.5457, 4.2644, 1.3800, 0.3446),
)


def grain_law_variance(response, channel, density):
    """Granularity variance at ``density`` for one record, in arbitrary units.

    Mirrors fotufilm's ``grainDensityProfile.variance`` and the three material laws it
    switches between.  Only the ratio to the anchor is ever used, so the scale cancels.
    """
    law = int(response["grain_law"])
    d = np.asarray(density, dtype=np.float64)
    if law == 1:                                    # silver: covered area, not density
        return d * 10.0 ** (0.21004 * d + 0.06114 * d * d)
    if law == 2:                                    # legacy single population: var ~ D
        return np.maximum(d, 0.0)
    if law == 3:                                    # dye reversal
        p, ds = (float(v) for v in np.asarray(response["grain_reversal"]))
        q = np.power(np.maximum(d, 1e-6) / max(ds, 1e-6), 2.0 * p)
        return q / (1.0 + q)
    a, toe, decay, hump, centre, width = (float(v)
                                          for v in np.asarray(response["grain_profile"])[channel])
    toe, decay, width = max(toe, 1e-4), max(decay, 1e-4), max(width, 1e-4)
    off = (d - centre) / width
    return (1.0 - np.exp(-d / toe)) * (
        1.0 + a * np.exp(-d / decay) + hump * np.exp(-0.5 * off * off))


def modulation_curve(response, channel, density):
    """Sigma multiplier at ``density`` relative to the datasheet read density.

    fotufilm's ``grainDensityModulation``.  This is a scalar function of density, so the
    shader does not evaluate it: :func:`bake_layer_table` resamples it onto the table's own
    axis.  An inline copy in the fragment shader made the 24 MP path return black frames
    without a single error -- the same scale ceiling that bit the grain field itself.

    A reversal with a measured curve (``response["reversal_gain"]``) returns that curve
    instead of the law: it was measured as the absolute target modulation, so multiplying the
    law by it would double-count.  ``docs/fotufilm-port-plan-2026-11-06.md`` §40.
    """
    if "reversal_gain" in response:
        span = np.asarray(response["reversal_span"], dtype=np.float64)
        t = np.asarray(density, dtype=np.float64) / max(float(span[channel]), 1e-6)
        return np.interp(np.clip(t, 0.0, 1.0),
                         np.asarray(response["reversal_grid"], dtype=np.float64),
                         np.asarray(response["reversal_gain"], dtype=np.float64)[channel])
    fog = float(response["grain_fog"])
    here = np.maximum(np.asarray(density, dtype=np.float64), 0.0) + fog
    anchor = max(float(np.asarray(response["grain_anchor"])[channel]), 0.0) + fog
    if anchor <= 1e-9:
        return np.ones_like(here)
    ratio = (grain_law_variance(response, channel, here)
             / max(float(grain_law_variance(response, channel, anchor)), 1e-20))
    return np.sqrt(np.maximum(ratio, 0.0))


def density_response(film_data, crystal=None, is_reversal=False, stock=None):
    """The per-record anchor density and the law that rescales sigma away from it.

    ``grainDensityModulation`` upstream divides the variance at the density in hand by the
    variance at the datasheet's read density and takes the square root -- the number is a
    sigma multiplier, so it must not touch the mean.  ``granularityReadDensity`` is gross for
    a reversal and net otherwise, which is the only place the two materials differ here.

    A reversal that has a **measured** curve (``core.grain_reversal_gain``) carries it instead
    of the law: the reference we compare against runs fotufilm's crystal simulation, and the
    analytic reversal law is a poor stand-in for it (§40).  The measured curve is absolute, so
    the law is replaced, not corrected -- see ``modulation_curve``.
    """
    curves = np.asarray(film_data.density_curves, dtype=np.float64)
    d_min = np.nanmin(curves, axis=0)
    d_max = np.nanmax(curves, axis=0)
    span = np.maximum(d_max - d_min, 1e-6)
    net = (1.0 - d_min) if is_reversal else np.ones(3)
    anchor = np.clip(net, 0.05 * span, 0.9 * span)

    crystal = crystal or {}
    try:
        law = int(crystal.get("density_law"))
    except (TypeError, ValueError):
        law = 0
    try:
        profile = np.asarray(crystal.get("density_profile"), dtype=np.float64)
        if profile.shape != (3, 6) or not np.isfinite(profile).all():
            raise ValueError(profile.shape)
    except (TypeError, ValueError):
        # `GrainDensityProfile(records:)` falls back to the family shape on anything malformed.
        profile = np.asarray(GRAIN_DENSITY_PROFILE_DEFAULT, dtype=np.float64)
    out = {
        "grain_law": law,
        "grain_profile": np.ascontiguousarray(profile, dtype=np.float32),
        "grain_anchor": anchor.astype(np.float32),
        "grain_fog": float(GRAIN_FOG_DENSITY),
        "grain_reversal": np.asarray(GRAIN_REVERSAL_PROFILE, dtype=np.float32),
    }
    if is_reversal and stock:
        from . import grain_reversal_gain
        curve = grain_reversal_gain.load(stock)
        if curve is not None:
            out["reversal_gain"] = np.ascontiguousarray(curve["mod"], dtype=np.float32)
            out["reversal_grid"] = np.ascontiguousarray(curve["grid"], dtype=np.float32)
            out["reversal_span"] = np.ascontiguousarray(curve["span"], dtype=np.float32)
    return out


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


def bake_layer_table(film_data, samples=256, crystal=None, is_reversal=False, stock=None):
    """Resample the sub-layer split onto a uniform density axis, ready for a 3-D texture.

    Returns ``dict`` with

    ``axis_min`` / ``axis_max``  (3,)  -- the density range each record is resampled over
    ``table``                    (samples, 3, 4) -- (layer0, layer1, layer2, modulation)

    Column order inside ``table`` matches ``density_curves_layers[:, sl, ch]``: the three
    coated layers of record ``ch``, fast -> slow.  The fourth column carries the stock's
    granularity-against-density multiplier on this same axis, so the shader gets the whole
    law for one component instead of an inlined branch tree.
    """
    axis = normalized_density_curves(film_data)                     # (M, 3)
    layers = np.asarray(film_data.density_curves_layers, dtype=np.float64)  # (M, 3, 3)

    lo = axis.min(axis=0)
    hi = axis.max(axis=0)
    uniform = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    table = np.zeros((samples, 3, 4), dtype=np.float32)
    response = density_response(film_data, crystal, is_reversal, stock)

    for ch in range(3):
        x = axis[:, ch]
        # np.interp needs an increasing axis; a stock with a flat record would break it.
        order = np.argsort(x)
        xs = x[order]
        for sl in range(3):
            ys = layers[order, sl, ch]
            values = np.interp(uniform * (hi[ch] - lo[ch]) + lo[ch], xs, ys)
            table[:, ch, sl] = values.astype(np.float32)
        table[:, ch, 3] = modulation_curve(
            response, ch, uniform * (hi[ch] - lo[ch]) + lo[ch]).astype(np.float32)

    return {
        "axis_min": lo.astype(np.float32),
        "axis_max": hi.astype(np.float32),
        "table": table,
    }


def grain_kernel_1d(pixel_size_um, final_blur_px=FRAME_BLUR_PX, radius=2,
                    decay_um=DYE_CLOUD_DECAY_UM, unsharp=None):
    """The discrete kernel the reference's dye-cloud cascade reduces to at this pitch.

    ``apply_grain_to_density_layers`` ends with two convolutions: the four Jarvis dye-cloud
    gaussians (weights ``JARVIS_DYE_CLOUD_GAUSSIANS``, sigmas scaled by ``decay_um/pitch``),
    then the frame blur (``FRAME_BLUR_PX``; upstream passes ``grain.blur`` = 0.65 there).
    Both are isotropic, so together they are one
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


def _read_density_rate(ceiling, density_min, uniformity):
    """Per-channel rate factor that pins sigma at the read density to ``strength*weights``.

    ``ceiling`` is ``density_max_abs.sum(0)`` -- the per-record ceiling *including* the
    record's share of the fog, which is what the shader divides the density by.

    The shader's variance is ``var = sum_l dmax_l^2 * p_l * sat_l / npp_l``.  Writing the
    layer split ``f_l`` (sum 1), with ``D = ceiling``, ``dmax_l = f_l * D``,
    ``npp_l = pixel_area * f_l * rate / (base*area_mul)`` and ``p_l = d / D`` with ``d`` the
    gross density in hand, the sum collapses -- ``sum_l f_l = 1`` -- to

        sigma^2 = D * d * sat * base * area_mul / (pixel_area * rate)

    so at the read density (``d`` is 1.0 gross for both materials: a negative reads at net 1,
    a reversal at net ``1 - Dmin`` plus its own ``Dmin``) the amplitude carries a ``sqrt(D)``
    the engine does not have.  Dividing the rate by ``D`` cancels it, and the saturation term
    ``sat = 1 - p*uniformity`` has to travel with it because ``p = 1/D`` moves with ``D``.

    What is left is ``sigma ~ strength * weight * sqrt(sat_ratio)`` -- the same three things
    fotufilm's ``readSigma = grainStrength * grainLayerWeights[l]`` names, and nothing about
    how deep the stock can print.  The anchor stock divides by itself, so its factor is
    exactly 1 and its render stays bit-identical (pinned, because a float32 round trip
    through ``density_max_layers`` leaves a 1e-8 residue that tips a few hundred pixels
    across a quantisation step).

    ``CRYSTAL_ANCHOR_TOTAL`` is the anchor stock's ceiling *before* its fog share, so the
    reference ``D0`` adds ``density_min`` back; ``ceiling`` arrives already carrying it.
    """
    total = np.asarray(ceiling, dtype=np.float64)
    dmin = np.asarray(density_min, dtype=np.float64)
    unif = np.asarray(uniformity, dtype=np.float64)
    span = np.maximum(total, 1e-6)
    span0 = np.maximum(np.asarray(CRYSTAL_ANCHOR_TOTAL, dtype=np.float64) + dmin, 1e-6)
    # Read density is 1.0 gross for both materials -- see the docstring.
    corr = (span / span0) * ((1.0 - unif / span) / (1.0 - unif / span0))
    return np.where(np.abs(corr - 1.0) < 1e-6, 1.0, corr)


def layer_scalars(film_data, grain, pixel_size_um, crystal=None, is_bw=False,
                  is_reversal=False, stock=None):
    """Per (record, sub-layer) numbers the shader needs besides the table.

    ``crystal`` is the stock's fotufilm archive dict (``size_mm`` / ``strength`` /
    ``layer_weights``) or ``None``.  Given it, the three gold-200 anchor constants resolve
    per stock (see ``CRYSTAL_ANCHOR``); ``None`` keeps the anchor constants themselves, and
    the anchor stock divides by its own fields, so both paths agree on it exactly.

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
    # Per-channel rate multipliers over the reference's own particle_scale -- see the
    # CHANNEL_RATE_SCALE calibration block above.  Dividing the area multiplies the rate;
    # the vector rides the channel axis of the (sublayer, channel) grid.
    if is_bw:
        # One silver image, one draw: the three records are the same emulsion seen three
        # times, so they get equal rates, and `colour_grain` 0 collapses the mix to the
        # shared field -- one fluctuation, which is what a single silver image is.
        #
        # The **kernel** stays the measured constant: mapping `size_mm` to the dye-cloud
        # decay predicts a lag-1 of 0.80 for these stocks against 0.556 / 0.561 measured, so
        # that mapping does not extend to the mono path.  The **amplitude** does follow the
        # archive though -- see the BW_ block above.  `_read_density_rate` takes the print's
        # own depth back out, exactly as it does for colour.
        decay_um = float(BW_DYE_CLOUD_DECAY_UM)
        rate = np.asarray(BW_RATE_SCALE, dtype=np.float64)
        area_mul = 1.0
        if crystal is not None:
            try:
                area_mul = (float(crystal["strength"]) / BW_ANCHOR_STRENGTH) ** 2
            except (KeyError, TypeError, ValueError):
                area_mul = 1.0
        rate = rate * _read_density_rate(density_max_abs.sum(axis=0), density_min,
                                         np.asarray(grain.uniformity))
    elif crystal is not None:
        try:
            decay_um = DYE_CLOUD_DECAY_UM * float(crystal["size_mm"]) / CRYSTAL_ANCHOR["size_mm"]
            weights = crystal.get("layer_weights")
            if not weights or len(weights) != 3:
                weights = CRYSTAL_ANCHOR["weights"]
            # `layer_weights` is an *amplitude* per record, not a particle count: fotufilm's
            # `readSigma = strength * grainLayerWeights[l]`, so a heavier weight means louder
            # grain.  sigma goes as 1/sqrt(rate), so the weight enters the rate inverted and
            # squared.  Reading it as a rate (the first wiring) had the sign backwards: gold
            # 200's blue record at 1.35 came out the *quietest* of the three.
            rate = np.asarray([CHANNEL_RATE_SCALE[i]
                               * (CRYSTAL_ANCHOR["weights"][i] / float(weights[i])) ** 2
                               for i in range(3)], dtype=np.float64)
            # Anchor the amplitude at the *read density*, the way fotufilm's `anchorToSheet`
            # does: solve the rate so sigma reads `strength * grainLayerWeights[l]`, with how
            # deep the stock can print cancelled out.  `_read_density_rate` carries the
            # derivation; the short version is that the sampled sigma goes as
            # ``sqrt((total+dmin) * d * sat / rate)``, so the ``sqrt(total)`` has to come out
            # of the rate and the saturation has to travel with it.
            rate = rate * _read_density_rate(density_max_abs.sum(axis=0), density_min,
                                             np.asarray(grain.uniformity))
            area_mul = (float(crystal["strength"]) / CRYSTAL_ANCHOR["strength"]) ** 2
        except (KeyError, TypeError, ValueError):
            decay_um = float(DYE_CLOUD_DECAY_UM)
            rate = np.asarray(CHANNEL_RATE_SCALE, dtype=np.float64)
            area_mul = 1.0
    else:
        decay_um = float(DYE_CLOUD_DECAY_UM)
        rate = np.asarray(CHANNEL_RATE_SCALE, dtype=np.float64)
        area_mul = 1.0
    particle_area = (float(grain.particle_area_um2) * area_mul
                     * np.asarray(grain.particle_scale, dtype=np.float64)[None, :]
                     * np.asarray(grain.particle_scale_layers, dtype=np.float64)[:, None]
                     / rate[None, :])
    particles = pixel_area * fractions / particle_area                  # (3 sl, 3 ch)

    od_particle = density_max_abs / np.maximum(particles, 1e-9)
    blur_sigma = float(grain.blur_dye_clouds_um) * np.sqrt(np.maximum(od_particle, 0.0))

    out = {
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
        # 0.5 is the reference's value -- half independent, half shared luminance; see the
        # calibration block above for why this is 1.0 here.
        "colour_grain": 0.0 if is_bw else COLOUR_GRAIN,
        "blur_sigma": blur_sigma.astype(np.float32),
        "blur_um": float(grain.blur_dye_clouds_um),
        # The Jarvis cascade's decay length.  Not in ``grain`` -- upstream hard-codes 1.45 in
        # `apply_grain_to_density_layers` -- so it travels with the bundle and both the CPU
        # reference and the shader's kernel read it from here.
        "dye_cloud_decay_um": decay_um,
        # Deliberately *not* ``grain.blur`` -- see ``FRAME_BLUR_PX``.  Keeping it here means
        # the CPU reference (grain_model) and the GPU path (gl_grain) both read the same value.
        "final_blur_px": FRAME_BLUR_PX,
        # Group-wide width of the dye-cloud cascade, folded into the shader's `grain_size`
        # slider.  See GROUP_KERNEL_SCALE: the two groups that carry their own rate/size
        # calibration come out ~11% coarser than the engine, colour negatives do not.
        "kernel_scale": float(_group_kernel_scale(is_bw, is_reversal)),
    }
    # Granularity against density: the law, its profile, and the per-record anchor it is
    # measured from.  Both renderers rescale sigma with it; the mean is untouched.
    out.update(density_response(film_data, crystal, is_reversal, stock))
    return out
