"""The preview's film grain: the same tiles, the same placement, on the GPU.

P2 gave the export a sampler that reads a stock's laid film (``core/film_grain``).  This module
gives the canvas the identical read, so the grain a photographer judges on screen is the grain
the export will write -- not a look-alike.  Everything that decides *which* film a pixel sees is
copied verbatim from ``core/film_grain/sampler.py``: the block hash, the eight turns, the seam
fade, the two layings, the level blend.  The only difference is how the box mean is taken: the
CPU walks a summed-area table in numpy, this reads the same table with the texture unit's
bilinear filter.

What travels to the GPU
-----------------------
``u_grain_sat``      ``(levels * 3, 257, 257)`` float32 -- each tile level's summed-area table,
                     mean-removed, one array layer per (record, level).
``u_grain_density``  the negative's developed density cube, ``(n, n, n)`` RGB = C, M, Y.  It rides
                     with the LUT (``canvas.set_3d_lut``) because the tile level a pixel reads is
                     its own density: indexing one bake's law with another bake's densities is
                     exactly the mistake this pairing prevents.
``u_grain_*``        the look: pitch, amount, size, colour, per-record weights, and the per-level
                     means / correlations the blend is renormalised by.

The pitch is the frame's own sampling -- texels of film per image pixel -- so the preview and the
export read the same film at the same scale regardless of how the canvas is zoomed.  Zooming out
only widens the pixel's box, which is what a downsampled read physically is.
"""
from __future__ import annotations

import numpy as np
from OpenGL import GL

__all__ = ["GLFilmGrain", "GRAIN_GLSL", "GRAIN_UNIFORM_NAMES", "MAX_LEVELS"]

MAX_LEVELS = 17                     # core/film_grain/tile_build.TILE_LEVELS
_TILE_SIDE = 256
_SAT_SIDE = _TILE_SIDE + 1

GRAIN_UNIFORM_NAMES = (
    "u_grain_sat", "u_grain_density", "u_grain_on", "u_grain_has_density",
    "u_grain_levels", "u_grain_mono", "u_grain_seed", "u_grain_dmin", "u_grain_dmax",
    "u_grain_pitch", "u_grain_size", "u_grain_softness", "u_grain_amount", "u_grain_colour",
    "u_grain_layer", "u_grain_means", "u_grain_rho", "u_grain_mean_light",
    "u_grain_image_size", "u_grain_probe",
)

# The stage itself.  ``GRAIN_GLSL`` is spliced into the fragment shader at the 3.6 slot; the
# functions below it mirror sampler.py line for line, and the comments name the CPU counterpart
# so the two can be diffed by eye.
GRAIN_GLSL = """
// ---- 3.6 film grain: the pixel is light through one small patch of the stock's film ----------
// Ported from fotufilm's film-tile model; the CPU twin is core/film_grain/sampler.py.
uniform sampler2DArray u_grain_sat;
uniform sampler3D u_grain_density;
uniform int   u_grain_on;
uniform int   u_grain_has_density;
uniform int   u_grain_levels;
uniform int   u_grain_mono;
uniform int   u_grain_seed;
uniform vec3  u_grain_dmin;
uniform vec3  u_grain_dmax;
uniform float u_grain_pitch;        // tile texels per image pixel, before Grain Size
uniform float u_grain_size;
uniform float u_grain_softness;
uniform float u_grain_amount;
uniform float u_grain_colour;
uniform vec3  u_grain_layer;
uniform float u_grain_means[51];        // -log10(level mean light), per (record, level)
uniform float u_grain_rho[51];          // neighbour correlation, per (record, level)
uniform float u_grain_mean_light[51];
uniform vec2  u_grain_image_size;
uniform int   u_grain_probe;        // diagnostic: write the density field instead of the frame

const float GRAIN_BLOCK  = 128.0;   // seam grid, in tile texels (1 texel = 1 um)
const float GRAIN_SEAM   = 16.0;
const float GRAIN_MARGIN = 64.0;
const float GRAIN_SIDE   = 256.0;
const float GRAIN_LOG10  = 2.302585092994046;

uint grain_pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A block's placement of the tile: an offset in texels and one of eight turns.
ivec3 grain_placement(int bx, int by, int laying, int record) {
    uint stream = uint(200 + 3 * laying + record) * 0x9E3779B9u;
    uint mixed = grain_pcg(uint(bx) ^ grain_pcg(uint(by)
                        ^ grain_pcg(uint(u_grain_seed) ^ stream)));
    return ivec3(int(mixed % 256u), int((mixed >> 8u) % 256u), int((mixed >> 16u) & 7u));
}

float grain_sat_lookup(int layer, vec2 p) {
    // Texel i of a 257-wide table sits at (i + 0.5) / 257, so a point p in [0, 256] lands
    // exactly between its two neighbours -- the same bilinear read the CPU does by hand.
    //
    // Two half-float channels, not one float: GL 3.0 requires linear filtering of half-float
    // textures but not of 32-bit float ones, and the table's sums reach ~3e4 where a half's own
    // step is 16 -- larger than the box difference being read.  Storing the value and its
    // residual restores the precision at no extra memory (2 x 16 bit = 32).
    vec2 texel = texture(u_grain_sat, vec3((p + 0.5) / 257.0, float(layer))).rg;
    return texel.x + texel.y;
}

float grain_sat(int layer, vec2 p) {
    // The tile repeats: a placement reaches up to one period past its far edge.
    vec2 over = step(vec2(GRAIN_SIDE), p);
    vec2 q = p - over * GRAIN_SIDE;
    float s = grain_sat_lookup(layer, q);
    s += over.x * grain_sat_lookup(layer, vec2(GRAIN_SIDE, q.y));
    s += over.y * grain_sat_lookup(layer, vec2(q.x, GRAIN_SIDE));
    s += over.x * over.y * grain_sat_lookup(layer, vec2(GRAIN_SIDE, GRAIN_SIDE));
    return s;
}

// Mean light through one pixel's box of film, held where the block's placement put it.
float grain_box_light(int record, int level, vec2 p0, vec2 p1, vec2 offset, float area) {
    int layer = record * u_grain_levels + level;
    vec2 a0 = p0 + offset;
    vec2 a1 = p1 + offset;
    float box = grain_sat(layer, vec2(a1.x, a1.y)) - grain_sat(layer, vec2(a0.x, a1.y))
              - grain_sat(layer, vec2(a1.x, a0.y)) + grain_sat(layer, vec2(a0.x, a0.y));
    return u_grain_mean_light[layer] + box / area;
}

// Block b's share of a point c: one inside, fading across the seam either side of an edge.
float grain_block_weight(float c, float b) {
    float start = b * GRAIN_BLOCK;
    return clamp((c - start + GRAIN_SEAM) / (2.0 * GRAIN_SEAM), 0.0, 1.0)
         * clamp((start + GRAIN_BLOCK + GRAIN_SEAM - c) / (2.0 * GRAIN_SEAM), 0.0, 1.0);
}

// Two levels' grain blended and renormalised by their correlation: blending two fields that are
// already alike would otherwise add variance and the mid-tone would grain harder than either end.
float grain_blend(float a, float b, float w, float rho) {
    float kept = (1.0 - w) * (1.0 - w) + w * w + 2.0 * w * (1.0 - w) * rho;
    return (a + (b - a) * w) / sqrt(max(kept, 1e-4));
}

// One record's density fluctuation at a pixel, in density units.
float grain_record(int record, float net, vec2 pixel, float footprint) {
    int levels = u_grain_levels;
    float span = max(u_grain_dmax[record] - u_grain_dmin[record], 1e-6);
    float t = clamp(net / span, 0.0, 1.0) * float(levels - 1);
    int k = clamp(int(floor(t)), 0, levels - 2);
    float u = t - float(k);
    int layer_a = record * levels + k;
    float half_box = 0.5 * footprint;
    float total = 0.0;

    for (int laying = 0; laying < 2; laying++) {
        // The layings' grids sit half a block apart; the box moves with the grid.
        float shift = GRAIN_BLOCK * float(laying) * 0.5;
        vec2 c = pixel + vec2(shift);
        vec2 lo = c - vec2(half_box);
        vec2 hi = c + vec2(half_box);
        int bx0 = int(floor((c.x - GRAIN_SEAM) / GRAIN_BLOCK));
        int bx1 = int(floor((c.x + GRAIN_SEAM) / GRAIN_BLOCK));
        int by0 = int(floor((c.y - GRAIN_SEAM) / GRAIN_BLOCK));
        int by1 = int(floor((c.y + GRAIN_SEAM) / GRAIN_BLOCK));

        float acc = 0.0;
        float norm = 0.0;
        // A pixel reaches its own block and, inside a seam band, the neighbour.  Where the two
        // indices agree the second candidate is the same block twice: it is skipped rather than
        // weighted, because a duplicated pass would scale the grain and its norm alike -- the
        // mean would still come out zero and only the variance would be wrong.
        for (int iy = 0; iy < 2; iy++) {
            for (int ix = 0; ix < 2; ix++) {
                if (ix == 1 && bx1 == bx0) continue;
                if (iy == 1 && by1 == by0) continue;
                int bx = (ix == 0) ? bx0 : bx1;
                int by = (iy == 0) ? by0 : by1;
                float w = grain_block_weight(c.x, float(bx)) * grain_block_weight(c.y, float(by));
                if (w <= 0.0) continue;

                ivec3 place = grain_placement(bx, by, laying, record);
                float left = float(bx) * GRAIN_BLOCK;
                float top = float(by) * GRAIN_BLOCK;
                // The placement reaches only so far past its block: clip the box to what it holds.
                float u0 = max(lo.x, left - GRAIN_MARGIN) - left;
                float u1 = min(hi.x, left + GRAIN_BLOCK + GRAIN_MARGIN) - left;
                float v0 = max(lo.y, top - GRAIN_MARGIN) - top;
                float v1 = min(hi.y, top + GRAIN_BLOCK + GRAIN_MARGIN) - top;
                float area = max((u1 - u0) * (v1 - v0), 1e-6);

                int turn = place.z;
                vec2 p0, p1;
                p0.x = ((turn & 1) != 0) ? GRAIN_BLOCK - u1 : u0;
                p1.x = ((turn & 1) != 0) ? GRAIN_BLOCK - u0 : u1;
                p0.y = ((turn & 2) != 0) ? GRAIN_BLOCK - v1 : v0;
                p1.y = ((turn & 2) != 0) ? GRAIN_BLOCK - v0 : v1;
                if ((turn & 4) != 0) {
                    float swap = p0.x; p0.x = p0.y; p0.y = swap;
                    swap = p1.x; p1.x = p1.y; p1.y = swap;
                }
                vec2 offset = vec2(GRAIN_MARGIN) + vec2(place.xy);

                float la = max(grain_box_light(record, k, p0, p1, offset, area), 1e-9);
                float lb = max(grain_box_light(record, k + 1, p0, p1, offset, area), 1e-9);
                float da = -log(la) / GRAIN_LOG10 - u_grain_means[layer_a];
                float db = -log(lb) / GRAIN_LOG10 - u_grain_means[layer_a + 1];
                acc += w * grain_blend(da, db, u, u_grain_rho[layer_a]);
                norm += w * w;
            }
        }
        total += acc / sqrt(max(norm, 1e-12));
    }
    return total * 0.7071067811865476;      // two independent fields of one variance: / sqrt(2)
}

// The negative's developed density per record, straight off the cube the LUT was baked with.
vec3 grain_net_density(vec3 linear) {
    if (u_grain_has_density == 1) {
        float dim = float(textureSize(u_grain_density, 0).x);
        vec3 idx = floor(clamp(linear_to_shaper(linear) * (dim - 1.0), vec3(0.0), vec3(dim - 1.0)));
        return texelFetch(u_grain_density, ivec3(idx), 0).rgb;
    }
    return -log(max(linear, vec3(1e-9))) / GRAIN_LOG10 - u_grain_dmin;
}

// The stage's entry point: density fluctuation for this pixel, in density units.
vec3 grain_field(vec3 linear, vec2 sample_uv, vec2 d_dx, vec2 d_dy) {
    float pitch = u_grain_pitch / max(u_grain_size, 0.01);
    // Pixel centres, in tile texels: the CPU reads (row + 0.5) * pitch, and this must be the
    // same number or the two samplers lay their blocks at different phases of the same film.
    vec2 pixel = sample_uv * u_grain_image_size * pitch;
    // How much film one fragment covers: at 1:1 it is one image pixel, and when the canvas is
    // zoomed out it is several -- a wider box, which is what a smaller read physically is.
    vec2 step_px = vec2(length(d_dx * u_grain_image_size), length(d_dy * u_grain_image_size));
    float cover = max(1.0, max(step_px.x, step_px.y));
    // A box wider than the seam grid can no longer be read out of one placement; the reference
    // never meets it, and past this point the grain is under a pixel anyway.
    float footprint = min(pitch * cover * max(u_grain_softness, 0.05), 192.0);
    vec3 net = grain_net_density(linear);

    if (u_grain_mono == 1) {
        float g = grain_record(1, net.g, pixel, footprint);
        return vec3(u_grain_amount * g);
    }
    vec3 g = vec3(grain_record(0, net.r, pixel, footprint),
                  grain_record(1, net.g, pixel, footprint),
                  grain_record(2, net.b, pixel, footprint));
    g *= u_grain_amount * u_grain_layer / u_grain_size;
    // Colour Grain: 1 keeps each record's own dye clouds, 0 shares one field across all three.
    float own = clamp(u_grain_colour, 0.0, 1.0);
    float shared = sqrt(max(3.0 - 2.0 * own * own, 0.0)) - own;
    return own * g + shared * vec3((g.r + g.g + g.b) / 3.0);
}
"""


class GLFilmGrain:
    """Owns one stock's tile textures and the uniforms that describe them."""

    def __init__(self) -> None:
        self._sat_tex = None
        self._density_tex = None
        self._density_size = 0
        self._density_ref = None
        self._layers = 0
        self._stock_id: str | None = None
        self._monochrome = False
        self._levels = 0
        self._means = None
        self._rho = None
        self._mean_light = None
        self._d_min = np.zeros(3)
        self._d_max = np.zeros(3)
        self._ready = False
        self._tiles = None
        self._arrays: dict = {}
        self._float_linear = True
        self.note = ""
        self._building: str | None = None
        self._on_ready = None       # called from the build thread; the canvas marshals it

    # -- tiles -------------------------------------------------------------------------------
    def ensure_stock(self, stock_id: str | None) -> bool:
        """Make the named stock's film resident.  Cheap and idempotent once it is."""
        if not stock_id or stock_id in (None, "none"):
            self._ready = False
            return False
        if self._ready and self._stock_id == stock_id:
            return True
        try:
            from core.film_grain import profiles, tiles_cache
            stock = profiles.stock_for(stock_id)
            if stock is None:
                self._ready = False
                self.note = "no baked grain for this stock"
                return False
            if self._stock_id != stock.stock_id or self._tiles is None:
                if not tiles_cache.is_cached(stock):
                    # Laying a film takes 20-40 s.  Doing it inside a paint looks like a hang,
                    # so it is started here and picked up by the next frame that finds it ready.
                    self._start_build(stock)
                    self._ready = False
                    self.note = f"laying {stock.stock_id}'s film..."
                    return False
                self._tiles = tiles_cache.load_or_build(stock)
            self._upload(self._tiles, stock)
            self._stock_id = stock.stock_id
            self._monochrome = bool(stock.is_monochrome)
            self._ready = True
            self.note = f"{stock.stock_id}: {self._levels} levels x 3 records"
            return True
        except Exception as exc:                     # noqa: BLE001 - a paint must never break
            print(f"[SpektraGrain] tile upload skipped: {exc}")
            self._ready = False
            self.note = str(exc)
            return False

    def _start_build(self, stock) -> None:
        if self._building == stock.stock_id:
            return
        self._building = stock.stock_id
        import threading

        def work():
            try:
                from core.film_grain import tiles_cache
                tiles_cache.load_or_build(stock)
            except Exception as exc:                 # noqa: BLE001 - reported, never raised
                print(f"[SpektraGrain] laying {stock.stock_id} failed: {exc}")
            finally:
                self._building = None
                if self._on_ready is not None:
                    self._on_ready()

        threading.Thread(target=work, daemon=True, name=f"grain-{stock.stock_id}").start()

    def set_ready_callback(self, callback) -> None:
        """Who to tell when a film finishes laying (emits a Qt signal, so any thread is fine)."""
        self._on_ready = callback

    def _upload(self, tiles, stock) -> None:
        levels = int(tiles.levels)
        self._levels = levels
        self._arrays = {}
        self._d_min = np.asarray(tiles.d_min, dtype=np.float64)
        self._d_max = np.asarray(tiles.d_max, dtype=np.float64)
        self._means = np.asarray(tiles.means(), dtype=np.float32)
        self._rho = np.asarray(tiles.correlations(), dtype=np.float32)
        self._mean_light = np.array([[tiles.mean_light(r, k) for k in range(levels)]
                                     for r in range(3)], dtype=np.float32)

        # Two half-float channels per texel: the value and its residual against half precision.
        # The sum of the two reads back at nearly float32 accuracy, through a texture format the
        # spec actually requires the driver to filter linearly.
        bank = np.empty((3 * levels, _SAT_SIDE, _SAT_SIDE, 2), dtype=np.float16)
        for record in range(3):
            for level in range(levels):
                table = tiles.summed_area(record, level).astype(np.float32)
                high = table.astype(np.float16)
                bank[record * levels + level, :, :, 0] = high
                bank[record * levels + level, :, :, 1] = (table - high.astype(np.float32))

        if self._sat_tex is None or self._layers != 3 * levels:
            if self._sat_tex:
                GL.glDeleteTextures([self._sat_tex])
            self._sat_tex = GL.glGenTextures(1)
            self._layers = 3 * levels
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self._sat_tex)
            GL.glTexImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, GL.GL_RG16F, _SAT_SIDE, _SAT_SIDE,
                            self._layers, 0, GL.GL_RG, GL.GL_HALF_FLOAT, bank)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self._sat_tex)
            GL.glTexSubImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, _SAT_SIDE, _SAT_SIDE,
                               self._layers, GL.GL_RG, GL.GL_HALF_FLOAT, bank)
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)

    # -- the density cube --------------------------------------------------------------------
    def set_density(self, cube) -> None:
        """Upload the negative-density cube that travelled with the LUT.

        Called on every paint: the identity check is what makes that free.  An offscreen render
        swaps in its own cube, and a stale one would index this bake's law with another's
        densities -- the failure this pairing exists to prevent.
        """
        if cube is None or cube is self._density_ref:
            return
        self._density_ref = cube
        data = np.ascontiguousarray(cube, dtype=np.float32)
        size = int(data.shape[0])
        if self._density_tex and self._density_size == size:
            GL.glBindTexture(GL.GL_TEXTURE_3D, self._density_tex)
            GL.glTexSubImage3D(GL.GL_TEXTURE_3D, 0, 0, 0, 0, size, size, size,
                               GL.GL_RGB, GL.GL_FLOAT, data)
            GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
            return
        if self._density_tex:
            GL.glDeleteTextures([self._density_tex])
        self._density_tex = GL.glGenTextures(1)
        self._density_size = size
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._density_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB32F, size, size, size, 0,
                        GL.GL_RGB, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

    # -- uniforms ----------------------------------------------------------------------------
    def bind(self, unit: int) -> int | None:
        """Bind the tile texture on ``unit``; returns the unit or ``None``."""
        if not self._ready or not self._sat_tex:
            return None
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self._sat_tex)
        return unit

    def density_unit(self, unit: int) -> int | None:
        if not self._density_tex:
            return None
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self._density_tex)
        return unit

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def monochrome(self) -> bool:
        return self._monochrome

    @property
    def has_density(self) -> bool:
        return self._density_tex is not None

    def scalars(self, params: dict, px_per_mm: float, image_size: tuple[int, int]) -> dict:
        """The float/int uniforms of one paint, as ``{name: value}``."""
        from core.film_grain import params_for
        look = params_for(params)
        pitch = (1.0 / max(px_per_mm, 1e-6)) / 0.001
        amount = float(look.amount)
        return {
            "u_grain_on": 1 if (self._ready and amount > 0.0) else 0,
            "u_grain_has_density": 1 if self._density_tex else 0,
            "u_grain_levels": int(self._levels),
            "u_grain_mono": 1 if self._monochrome else 0,
            "u_grain_seed": 1,
            "u_grain_dmin": tuple(float(v) for v in self._d_min),
            "u_grain_dmax": tuple(float(v) for v in self._d_max),
            "u_grain_pitch": float(pitch),
            "u_grain_size": float(look.size),
            "u_grain_softness": float(look.scan_softness),
            "u_grain_amount": amount,
            "u_grain_colour": float(look.colour),
            "u_grain_layer": tuple(float(v) for v in look.layers),
            "u_grain_image_size": (float(image_size[0]), float(image_size[1])),
            "u_grain_probe": 1 if params.get("grain_probe") else 0,
        }

    def arrays(self) -> dict:
        """The array uniforms, already flattened and padded to the shader's fixed size."""
        if not self._ready or self._means is None:
            return {}
        if self._arrays:
            return self._arrays
        means = np.zeros(3 * MAX_LEVELS, dtype=np.float32)
        rho = np.zeros(3 * MAX_LEVELS, dtype=np.float32)
        light = np.zeros(3 * MAX_LEVELS, dtype=np.float32)
        levels = self._levels
        for record in range(3):
            base = record * MAX_LEVELS
            means[base:base + levels] = self._means[record][:levels]
            light[base:base + levels] = self._mean_light[record][:levels]
            count = max(levels - 1, 0)
            rho[base:base + count] = self._rho[record][:count]
        self._arrays = {"u_grain_means": means, "u_grain_rho": rho,
                        "u_grain_mean_light": light}
        return self._arrays

    def release(self) -> None:
        ids = [t for t in (self._sat_tex, self._density_tex) if t]
        if ids:
            try:
                GL.glDeleteTextures(ids)
            except Exception:                        # noqa: BLE001 - context may be gone
                pass
        self._sat_tex = None
        self._density_tex = None
        self._ready = False
