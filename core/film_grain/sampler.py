"""The CPU sampler: a frame's pixels read the stock's film through their own small windows.

Ported from ``FilmGrainTiles.applyTiled`` (and the kernel it mirrors, ``Stages/FilmTiles.h``).

What a pixel does
-----------------
1. Its gross density picks the two tile levels either side of it, ``k`` and ``k + 1``, and the
   weight ``w`` between them.
2. It reads the light through its **box of film** -- ``footprint`` texels a side, centred on the
   pixel -- out of the level's summed-area table, four lookups per corner, exact at any pitch.
3. The frame is cut into 128 µm blocks; each block reads the tile at its own hashed offset and one
   of eight flips and turns, and across a 16 µm seam either side of a block edge the neighbouring
   placements cross-fade.  A placement reaches 64 µm past its block, so a cloud near an edge is
   never cut along a straight line.
4. One 256 µm tile laid once would recur whole every 0.7 mm, so every pixel reads **two
   independent layings**, the second on a grid offset by half a block, and their grain is summed
   and divided by √2: independent fields of one covariance keep it.
5. Each read is ``-log10(light)`` less the level's mean density, blended between the two levels
   and renormalised by their correlation so the blend keeps its variance.

The result is a density fluctuation ``dD``, added to the negative's density: the print sees the
grain the way it sees the film's own, spread by the enlarger's spot.
"""
from __future__ import annotations

import numpy as np

from . import tile_build as tb

__all__ = ["TileSampler", "block_placement", "look_geometry"]

TILE_LAYERS = 2                     # independent layings of the same tile
BLOCK_TEXELS = 128
SEAM_TEXELS = 16
MARGIN_TEXELS = 64


def look_geometry(px_per_mm: float, size: float = 1.0, softness: float = 1.0) -> tuple[float, float]:
    """Pixel pitch and reading footprint, in tile texels.

    ``size`` magnifies the film under the frame; the grain grows with it and its fluctuation falls
    by as much, so the 48 µm aperture still reads the sheet.  ``softness`` is the side of the
    square of film each pixel reads, in pixels: wider averages the grain down as a softer scan
    does.
    """
    size = min(max(size, 0.5), 4.0)
    softness = min(max(softness, 0.5), 4.0)
    pitch = (1.0 / max(px_per_mm, 1e-6)) / tb.TILE_TEXEL_MM / size
    return pitch, pitch * softness


def _pcg_hash(value: np.ndarray) -> np.ndarray:
    """The reference's uint32 hash; the wraps are the point of it, not an accident."""
    with np.errstate(over="ignore"):
        state = (value.astype(np.uint32) * np.uint32(747796405)
                 + np.uint32(2891336453)).astype(np.uint32)
        word = ((state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state).astype(np.uint32)
        word = (word * np.uint32(277803737)).astype(np.uint32)
    return ((word >> np.uint32(22)) ^ word).astype(np.uint32)


def block_placement(seed: int, layer: int, record: int, bx: np.ndarray,
                    by: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A block's placement of the tile: offset in texels and one of eight orientations."""
    with np.errstate(over="ignore"):
        stream = (np.uint32((200 + 3 * layer + record) & 0xFFFFFFFF)
                  * np.uint32(0x9E3779B9))
    mixed = _pcg_hash(bx.astype(np.uint32) ^ _pcg_hash(by.astype(np.uint32)
                                                      ^ _pcg_hash(np.uint32(seed & 0xFFFFFFFF) ^ stream)))
    side = np.uint32(256)
    return ((mixed % side).astype(np.float64), ((mixed >> np.uint32(8)) % side).astype(np.float64),
            ((mixed >> np.uint32(16)) & np.uint32(7)).astype(np.int64))


def _block_weight(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Block ``b``'s share of a point ``c`` texels along one axis: 1 inside, fading at the seam."""
    start = b * float(BLOCK_TEXELS)
    return (np.clip((c - start + SEAM_TEXELS) / (2.0 * SEAM_TEXELS), 0.0, 1.0)
            * np.clip((start + BLOCK_TEXELS + SEAM_TEXELS - c) / (2.0 * SEAM_TEXELS), 0.0, 1.0))


def _sum_at(bank: np.ndarray, level: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear read of the summed-area table at a real point (clamped inside one period)."""
    n = bank.shape[1] - 1
    ix = np.clip(x.astype(np.int64), 0, n - 1)
    iy = np.clip(y.astype(np.int64), 0, n - 1)
    ax = x - ix
    ay = y - iy
    s00 = bank[level, iy, ix]
    s10 = bank[level, iy, ix + 1]
    s01 = bank[level, iy + 1, ix]
    s11 = bank[level, iy + 1, ix + 1]
    return s00 + (s10 - s00) * ax + (s01 - s00) * ay + (s11 - s10 - s01 + s00) * ax * ay


def _wrapped_sum_at(bank: np.ndarray, level: np.ndarray, x: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
    """The same, up to a period past the tile's far edge: the tile repeats."""
    n = bank.shape[1] - 1
    over_x = x >= n
    over_y = y >= n
    if not (over_x.any() or over_y.any()):
        return _sum_at(bank, level, x, y)
    qx = np.where(over_x, x - n, x)
    qy = np.where(over_y, y - n, y)
    total = _sum_at(bank, level, qx, qy)
    if over_x.any():
        total = total + np.where(over_x, _sum_at(bank, level, np.full_like(qx, float(n)), qy), 0.0)
    if over_y.any():
        total = total + np.where(over_y, _sum_at(bank, level, qx, np.full_like(qy, float(n))), 0.0)
    if (over_x & over_y).any():
        total = total + np.where(over_x & over_y, bank[level, n, n], 0.0)
    return total


def _blend(a: np.ndarray, b: np.ndarray, w: np.ndarray, rho: float) -> np.ndarray:
    kept = (1.0 - w) ** 2 + w * w + 2.0 * w * (1.0 - w) * rho
    return (a + (b - a) * w) / np.sqrt(np.maximum(kept, 1e-4))


class TileSampler:
    """Reads one stock's tiles for frames at one pitch."""

    def __init__(self, tiles, px_per_mm: float, seed: int = 1, size: float = 1.0,
                 softness: float = 1.0) -> None:
        self.tiles = tiles
        self.seed = int(seed)
        self.size = min(max(size, 0.5), 4.0)
        self.pitch, self.footprint = look_geometry(px_per_mm, size, softness)
        self.means = tiles.means()
        self.rho = tiles.correlations()
        self.sigmas = tiles.sigmas()
        self.d_min = np.asarray(tiles.d_min, dtype=np.float64)
        self.d_max = np.asarray(tiles.d_max, dtype=np.float64)
        self.levels = int(tiles.levels)
        self.mean_lights = np.asarray(
            [[tiles.mean_light(r, k) for k in range(tiles.levels)] for r in range(3)],
            dtype=np.float64)
        self._banks: dict[int, np.ndarray] = {}

    # -- lazily stacked summed-area tables ---------------------------------------------------
    def bank(self, record: int) -> np.ndarray:
        """The record's levels' summed-area tables, stacked so one gather can pick a level."""
        table = self._banks.get(record)
        if table is None:
            table = np.stack([self.tiles.summed_area(record, k)
                              for k in range(self.tiles.levels)]).astype(np.float32)
            self._banks[record] = table
        return table

    # -- one record -------------------------------------------------------------------------
    def _record(self, record: int, net: np.ndarray, row_offset: int) -> np.ndarray:
        tiles = self.tiles
        height, width = net.shape
        levels = tiles.levels
        lo = float(tiles.d_min[record])
        hi = float(tiles.d_max[record])

        # Gross density is the base plus what the pipeline developed; the levels run D-min..D-max.
        t = np.clip(net / max(hi - lo, 1e-6), 0.0, 1.0) * (levels - 1)
        k = np.clip(np.floor(t).astype(np.int64), 0, levels - 2).ravel()
        u = (t - k.reshape(height, width)).ravel()
        del t

        half = 0.5 * self.footprint
        pixel_x = ((np.arange(width) + 0.5) * self.pitch)[None, :]
        pixel_y = ((row_offset + np.arange(height) + 0.5) * self.pitch)[:, None]

        grain = np.zeros(height * width, dtype=np.float64)
        for laying in range(TILE_LAYERS):
            # The laying's grid sits half a block from the frame's; its box moves with it.
            shift = BLOCK_TEXELS * laying / TILE_LAYERS
            cx = (pixel_x + shift)
            cy = (pixel_y + shift)
            boxes = (np.broadcast_to(cx - half, (height, width)).ravel(),
                     np.broadcast_to(cx + half, (height, width)).ravel(),
                     np.broadcast_to(cy - half, (height, width)).ravel(),
                     np.broadcast_to(cy + half, (height, width)).ravel())
            bx0 = np.floor((cx - SEAM_TEXELS) / BLOCK_TEXELS).astype(np.int64)
            bx1 = np.floor((cx + SEAM_TEXELS) / BLOCK_TEXELS).astype(np.int64)
            by0 = np.floor((cy - SEAM_TEXELS) / BLOCK_TEXELS).astype(np.int64)
            by1 = np.floor((cy + SEAM_TEXELS) / BLOCK_TEXELS).astype(np.int64)
            grid = self._placements(record, laying, bx0, bx1, by0, by1)
            extra_x = (bx1 != bx0)
            extra_y = (by1 != by0)

            laying_grain = np.zeros(height * width, dtype=np.float64)
            norm = np.zeros(height * width, dtype=np.float64)
            # A pixel's fade reaches the block it sits in and, inside a seam band, the neighbour.
            # The two indices are equal away from the seams, so each candidate is counted once: a
            # duplicated pass would scale the grain and its norm alike -- the mean would still
            # come out zero and only the variance would be wrong.
            for bx, by, wanted in ((bx0, by0, None),
                                   (bx0, by1, extra_y),
                                   (bx1, by0, extra_x),
                                   (bx1, by1, extra_x & extra_y)):
                weight = _block_weight(cx, bx) * _block_weight(cy, by)
                if wanted is not None:
                    weight = np.where(wanted, weight, 0.0)
                weight = np.broadcast_to(weight, (height, width)).ravel()
                if wanted is None:
                    select = slice(None)
                else:
                    index = np.flatnonzero(weight)
                    if index.size == 0:
                        continue
                    select = index
                self._read(record, select, grid, k, u, boxes,
                           np.broadcast_to(bx, (height, width)).ravel(),
                           np.broadcast_to(by, (height, width)).ravel(),
                           weight, laying_grain, norm)
            grain += laying_grain / np.sqrt(np.maximum(norm, 1e-12))
        return (grain / np.sqrt(TILE_LAYERS)).reshape(height, width)

    def _placements(self, record: int, laying: int, bx0, bx1, by0, by1):
        """Every block's placement over the chunk's block range, computed once per laying."""
        x_lo, x_hi = int(bx0.min()), int(bx1.max())
        y_lo, y_hi = int(by0.min()), int(by1.max())
        xs = np.arange(x_lo, x_hi + 1)
        ys = np.arange(y_lo, y_hi + 1)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        offset_x, offset_y, turn = block_placement(self.seed, laying, record, grid_x, grid_y)
        return (x_lo, y_lo, offset_x.astype(np.float32), offset_y.astype(np.float32), turn)

    def _read(self, record, select, grid, k, u, boxes, bx, by, weight, grain, norm) -> None:
        bank = self.bank(record)
        x_lo, y_lo, off_x, off_y, turn_grid = grid

        x0, x1, y0, y1 = boxes
        if isinstance(select, slice):
            kk, uu, wx0, wx1, wy0, wy1, bxs, bys = k, u, x0, x1, y0, y1, bx, by
        else:
            kk, uu = k[select], u[select]
            wx0, wx1, wy0, wy1 = x0[select], x1[select], y0[select], y1[select]
            bxs, bys = bx[select], by[select]

        offset_x = off_x[bxs - x_lo, bys - y_lo]
        offset_y = off_y[bxs - x_lo, bys - y_lo]
        turn = turn_grid[bxs - x_lo, bys - y_lo]
        left = (bxs * float(BLOCK_TEXELS)).astype(np.float32)
        top = (bys * float(BLOCK_TEXELS)).astype(np.float32)
        # The placement reaches only so far past its block: clip the box to what it holds.
        u0 = np.maximum(wx0, left - MARGIN_TEXELS) - left
        u1 = np.minimum(wx1, left + BLOCK_TEXELS + MARGIN_TEXELS) - left
        v0 = np.maximum(wy0, top - MARGIN_TEXELS) - top
        v1 = np.minimum(wy1, top + BLOCK_TEXELS + MARGIN_TEXELS) - top
        area = np.maximum((u1 - u0) * (v1 - v0), 1e-6)

        flip_x = (turn & 1) != 0
        flip_y = (turn & 2) != 0
        swap = (turn & 4) != 0
        p0 = np.where(flip_x, BLOCK_TEXELS - u1, u0).astype(np.float32)
        p1 = np.where(flip_x, BLOCK_TEXELS - u0, u1).astype(np.float32)
        q0 = np.where(flip_y, BLOCK_TEXELS - v1, v0).astype(np.float32)
        q1 = np.where(flip_y, BLOCK_TEXELS - v0, v1).astype(np.float32)
        p0, q0 = np.where(swap, q0, p0), np.where(swap, p0, q0)
        p1, q1 = np.where(swap, q1, p1), np.where(swap, p1, q1)
        sx = (MARGIN_TEXELS + offset_x).astype(np.float32)
        sy = (MARGIN_TEXELS + offset_y).astype(np.float32)

        level_a = kk
        light_a = self._box_light(bank, record, level_a, p0, p1, q0, q1, sx, sy, area)
        light_b = self._box_light(bank, record, level_a + 1, p0, p1, q0, q1, sx, sy, area)
        da = -np.log10(np.maximum(light_a, 1e-9)) - self.means[record][level_a]
        db = -np.log10(np.maximum(light_b, 1e-9)) - self.means[record][level_a + 1]
        blended = _blend(da, db, uu, self.rho[record][level_a])
        grain[select] += weight[select] * blended
        norm[select] += weight[select] * weight[select]

    def _box_light(self, bank, record, level, p0, p1, q0, q1, sx, sy, area) -> np.ndarray:
        """Mean light through each pixel's box, as the block's placement holds the tile."""
        ax0, ax1 = p0 + sx, p1 + sx
        ay0, ay1 = q0 + sy, q1 + sy
        box = (_wrapped_sum_at(bank, level, ax1, ay1) - _wrapped_sum_at(bank, level, ax0, ay1)
               - _wrapped_sum_at(bank, level, ax1, ay0) + _wrapped_sum_at(bank, level, ax0, ay0))
        return self.mean_lights[record][level] + box / area

    # -- the frame --------------------------------------------------------------------------
    def apply(self, net_density: np.ndarray, row_offset: int, amount: float, colour: float = 1.0,
              layer_weights=(1.0, 1.0, 1.0), monochrome: bool = False,
              luma_correlation: float = 0.0) -> np.ndarray:
        """Density fluctuation for a chunk of the frame, ``(height, width, 3)``.

        ``net_density`` is the negative's developed density above base per record, in the stock's
        own units -- the quantity the tile levels are indexed by.

        ``luma_correlation`` ties the three records' dye clouds to one common field, the way a
        real coating's layers are only partly independent.  The reference packs it as
        ``grainLumaCorrelation`` and documents the invariant we honour here: the *spread* must not
        move with the correlation, only how much of it the three channels share.  Mixing in the
        scaled mean keeps both true -- variance stays ``sigma^2`` while the cross-correlation of
        any two records becomes exactly ``rho``.
        """
        height, width = net_density.shape[:2]
        if monochrome:
            grain = self._record(1, net_density[..., 1], row_offset)
            return np.repeat((amount * grain)[..., None], 3, axis=2)
        raw = np.empty((height, width, 3), dtype=np.float64)
        for record in range(3):
            raw[..., record] = self._record(record, net_density[..., record], row_offset)

        own = min(max(colour, 0.0), 1.0)
        rho = min(max(float(luma_correlation), 0.0), 1.0)
        # At Colour Grain 0 the three channels are already one field, so there is no independence
        # left to couple -- mixing there only rescales it.
        if rho > 0.0 and own > 0.05:
            raw = self._couple_records(raw, net_density, min(rho, 0.99))

        grain = raw * (amount * np.asarray([max(float(w), 0.0) for w in layer_weights],
                                           dtype=np.float64) / self.size)

        shared = np.sqrt(max(3.0 - 2.0 * own * own, 0.0)) - own
        if rho > 0.0 and own > 0.05:
            # The coefficient above assumes independent records; once they are coupled the mean is
            # wider by sqrt((1 + 2 rho) / 3), so divide that back out.
            shared /= np.sqrt(1.0 + 2.0 * min(rho, 0.99))
        mean = grain.mean(axis=2, keepdims=True)
        return own * grain + shared * mean

    # -- luma correlation --------------------------------------------------------------------
    def _sigma_at(self, record: int, net: np.ndarray) -> np.ndarray:
        """The record's own spread at each pixel's density, ``(h, w)``.

        Interpolated over the level table rather than taken as one number: the grain is coarser
        in the dense end, and the records do not sit at the same density for the same pixel.
        """
        span = max(float(self.d_max[record] - self.d_min[record]), 1e-6)
        x = np.clip((net - self.d_min[record]) / span * (self.levels - 1), 0.0, self.levels - 1.0)
        table = self.sigmas[record]
        lo = np.floor(x).astype(np.int64)
        hi = np.minimum(lo + 1, self.levels - 1)
        w = x - lo
        return np.maximum((1.0 - w) * table[lo] + w * table[hi], 1e-6)

    def _couple_records(self, raw: np.ndarray, net_density: np.ndarray, rho: float) -> np.ndarray:
        """Tie the three records' fluctuations to one field, without changing their spread.

        The target is the reference's invariant: ``grainLumaCorrelation`` sets how much of the
        grain the channels *share*, and the granularity must not move with it (their
        LuminanceChannelTests asserts the ratio is constant from rho 0 to rho 1).

        Work in units of each record's own sigma -- the records differ by up to 1.6x, so a plain
        mix would rescale the coarser one -- then, with ``u`` the standardised records and ``s``
        the standardised mean (which each ``u`` is correlated with at 1/sqrt(3)):

            out = (u + c s) / sqrt(1 + a),   a = 2 c / sqrt(3) + c^2

        Picking ``a = rho / (1 - rho)`` makes the cross-correlation exactly ``rho`` and leaves the
        variance at one, for every rho and every record.
        """
        sigma = np.stack([self._sigma_at(r, net_density[..., r]) for r in range(3)], axis=2)
        unit = raw / sigma
        shared = np.sqrt(3.0) * unit.mean(axis=2, keepdims=True)
        gamma = 1.0 / np.sqrt(3.0)
        target = rho / (1.0 - rho)
        c = -gamma + np.sqrt(gamma * gamma + target)
        return (raw + c * sigma * shared) / np.sqrt(1.0 + target)
