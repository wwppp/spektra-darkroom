"""Each stock's film, laid once and kept: the tile bank the sampler reads.

A tile is 256 µm of film at 1 µm texels, at 17 gross densities from D-min to D-max, per record.
Laying one level costs 0.4–1.1 s, so a record is 7–19 s and a stock half a minute -- once per
stock, ever, if it is kept.  That is what this module does: build, anchor, and write to the user
cache.

The cache key carries everything that can change the film: the stock's baked grain parameters,
the tile version, the level count and the seed.  Change any of them and the tiles are laid again;
nothing else in the application needs to know.

What is stored
--------------
``lights``   float16, ``(3, levels, 256, 256)`` -- the transmittance through each level.  Half
             precision is 0.05 % of a transmission step, far below the 8-bit output's 0.4 %, and
             halves the file (a stock is ~13 MB at float32).
``sat``      not stored: the summed-area tables are one cumulative sum away and would be 4x the
             size for no information.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import tile_build as tb
from .population import CrystalPopulation, GrainStock

__all__ = ["StockTiles", "build_tiles", "load_or_build", "is_cached", "TileStore", "signature_of"]

CACHE_VERSION = 6          # 6: anchor keeps its closest pass (sigma is not monotone in dye)


def _cache_root() -> Path:
    from core.path_utils import get_user_cache_dir
    root = Path(get_user_cache_dir()) / "film_grain"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _signature(stock: GrainStock, levels: int, seed: int) -> str:
    payload = {
        "version": CACHE_VERSION,
        "levels": levels,
        "seed": seed,
        "size": stock.grain_size_mm,
        "strength": stock.grain_strength,
        "weights": list(stock.grain_layer_weights),
        "ratios": list(stock.grain_layer_size_ratio),
        "population": [stock.population.radius_span, list(stock.population.sublayer_shares),
                       stock.population.coating_density_scale],
        "silver": stock.is_silver,
        "reversal": stock.is_reversal,
        "fog": stock.grain_fog_density,
        "curves": [[round(float(c.density(x)), 6) for x in (-4.0, -2.0, 0.0, 2.0, 4.0)]
                   for c in stock.curves],
    }
    # Only a pushed development carries a non-unit scale.  Keying unconditionally would
    # invalidate every standard film on disk for a value that is always 1.0 there.
    if any(abs(float(s) - 1.0) > 1e-9 for s in stock.anchor_scales):
        payload["anchor"] = [round(float(s), 6) for s in stock.anchor_scales]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


@dataclass
class StockTiles:
    """A stock's laid film: the light through each level, per record."""

    stock_id: str
    d_min: np.ndarray                  # (3,)
    d_max: np.ndarray                  # (3,)
    lights: np.ndarray                 # (3, levels, side, side) float32
    active: tuple[int, ...]            # records that carry grain

    # Content hash of everything that shaped this film (see ``_signature``).  Set by the store
    # that handed the object out; two developments of one stock share ``stock_id`` but not this,
    # so anything caching per-stock objects must key on it as well.
    signature: str = ""

    _sat: dict = field(default_factory=dict, repr=False)
    _means: np.ndarray | None = field(default=None, repr=False)
    _rho: np.ndarray | None = field(default=None, repr=False)
    _sigma: np.ndarray | None = field(default=None, repr=False)

    @property
    def levels(self) -> int:
        return int(self.lights.shape[1])

    @property
    def side(self) -> int:
        return int(self.lights.shape[2])

    def light(self, record: int, level: int) -> np.ndarray:
        return self.lights[record, level]

    def summed_area(self, record: int, level: int) -> np.ndarray:
        """``(side+1)²`` running sum of ``light - level mean``: any box is four lookups."""
        key = (record, level)
        table = self._sat.get(key)
        if table is None:
            light = self.light(record, level)
            shifted = light - float(light.mean())
            table = np.zeros((self.side + 1, self.side + 1), dtype=np.float32)
            table[1:, 1:] = shifted.cumsum(axis=0, dtype=np.float64).cumsum(
                axis=1, dtype=np.float64).astype(np.float32)
            self._sat[key] = table
        return table

    def mean_light(self, record: int, level: int) -> float:
        return float(self.light(record, level).mean())

    def means(self) -> np.ndarray:
        """Mean density of each level, ``(3, levels)``: what the blend is measured about.

        Every record, not only the active ones: a monochrome stock lays one film and copies it to
        all three, and a caller reading record 0 must not find ``nan``.
        """
        if self._means is None:
            means = np.full((3, self.levels), np.nan, dtype=np.float64)
            for record in range(3):
                for level in range(self.levels):
                    means[record, level] = -np.log10(max(self.mean_light(record, level), 1e-9))
            self._means = means
        return self._means

    def sigmas(self) -> np.ndarray:
        """Spread of the density fluctuation at each level, ``(3, levels)``.

        The records do not carry the same spread -- a record with a wider density range has more
        grain -- so tying them to one common field has to be done in units of each record's own
        sigma, or the mix silently rescales the coarser record.  That is what this table is for.
        """
        if self._sigma is None:
            sigma = np.full((3, self.levels), np.nan, dtype=np.float64)
            for record in range(3):
                for level in range(self.levels):
                    density = -np.log10(np.maximum(self.light(record, level), 1e-9))
                    sigma[record, level] = float(density.std())
            self._sigma = sigma
        return self._sigma

    def correlations(self) -> np.ndarray:
        """Correlation of neighbouring levels' grain, ``(3, levels-1)``.

        A pixel whose density falls between two levels blends their grain; blending two fields
        that are already alike adds variance, so the blend is renormalised by this.
        """
        if self._rho is None:
            rho = np.zeros((3, max(self.levels - 1, 1)), dtype=np.float64)
            for record in range(3):
                densities = [-np.log10(np.maximum(self.light(record, k), 1e-9))
                             for k in range(self.levels)]
                for k in range(self.levels - 1):
                    a = densities[k].ravel()
                    b = densities[k + 1].ravel()
                    a = a - a.mean()
                    b = b - b.mean()
                    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
                    rho[record, k] = float(np.sum(a * b) / denominator) if denominator else 0.0
            self._rho = rho
        return self._rho


def _anchor(record: tb.FilmRecord, model: CrystalPopulation, scale: float = 1.0) -> tb.FilmRecord:
    """Re-solve the dye until the tile reads the sheet's granularity at the read density.

    The read density is a *net* density -- one above base -- so the gross density to lay is
    simply ``d_min + read_net``.  Going through ``log_exposure_for`` and back would put a
    reversal on the far end of its ladder (net 1.0 → gross 2.42 on Provia 100F), and the anchor
    would then be taken somewhere the sheet never measured.

    ``scale`` carries a pushed development's gain: the sheet's figure belongs to the reference
    process, and ``GrainStock.anchor_scales`` says how much coarser this development is there.
    """
    gross = float(record.d_min + model.read_net_density)
    return tb.anchor_to_sheet(record, gross, model.read_sigma * float(scale))


def build_tiles(stock: GrainStock, levels: int = tb.TILE_LEVELS, seed: int = 1,
                verbose: bool = False) -> StockTiles:
    """Lay and anchor every record of ``stock``, then lay all its levels."""
    d_min = np.zeros(3)
    d_max = np.zeros(3)
    lights = np.zeros((3, levels, tb.TILE_SIDE, tb.TILE_SIDE), dtype=np.float32)
    active = [1] if stock.is_monochrome else [0, 1, 2]
    reference = None
    if stock.reference_id:
        # A pushed development: the coating comes from the standard one, the curve from the
        # condition, and ``refitDevelopment`` bridges them.
        from . import profiles
        reference = profiles.stock_for(stock.reference_id, 0)
    for record in active:
        started = time.time()
        model = CrystalPopulation(stock, record, reference=reference)
        laid = tb.sublayers_for(model)
        scale = float(stock.anchor_scales[record]) if record < len(stock.anchor_scales) else 1.0
        anchored = _anchor(laid, model, scale)
        d_min[record], d_max[record] = anchored.d_min, anchored.d_max
        for level, gross in enumerate(anchored.grosses(levels)):
            lights[record, level] = tb.build_level(anchored, gross, seed=seed).astype(np.float32)
        if verbose:
            print(f"  {stock.stock_id} record {record}: {time.time() - started:.1f} s")
    if stock.is_monochrome:
        lights[0] = lights[1]
        lights[2] = lights[1]
        d_min[0] = d_min[2] = d_min[1]
        d_max[0] = d_max[2] = d_max[1]
    return StockTiles(stock_id=stock.stock_id, d_min=d_min, d_max=d_max, lights=lights,
                      active=tuple(active))


def _path(stock: GrainStock, levels: int, seed: int) -> Path:
    return _cache_root() / f"{stock.stock_id}-{_signature(stock, levels, seed)}.npz"


def signature_of(stock: GrainStock, levels: int = tb.TILE_LEVELS, seed: int = 1) -> str:
    """The content hash that names this film on disk and in memory.

    Exposed because a caller holding tiles has to tell two developments of one stock apart
    without re-deriving the curves it cannot see.
    """
    return _signature(stock, levels, seed)


def is_cached(stock: GrainStock, levels: int = tb.TILE_LEVELS, seed: int = 1) -> bool:
    """Whether this stock's film is already on disk.

    The preview asks before it reads: laying a stock costs twenty to forty seconds, and a paint
    that blocks for that long is indistinguishable from a hang.  Callers that cannot wait check
    this first and lay the film off the paint thread.
    """
    return _path(stock, levels, seed).exists()


def load_or_build(stock: GrainStock, levels: int = tb.TILE_LEVELS, seed: int = 1,
                  verbose: bool = False) -> StockTiles:
    """Read the stock's tiles from the cache, laying them first if they are not there."""
    path = _path(stock, levels, seed)
    if path.exists():
        try:
            with np.load(path) as data:
                return StockTiles(stock_id=stock.stock_id, d_min=data["d_min"],
                                  d_max=data["d_max"],
                                  lights=data["lights"].astype(np.float32),
                                  active=tuple(int(r) for r in data["active"]))
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
    tiles = build_tiles(stock, levels=levels, seed=seed, verbose=verbose)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, d_min=tiles.d_min, d_max=tiles.d_max, lights=tiles.lights.astype(np.float16),
             active=np.asarray(tiles.active, dtype=np.int16))
    os.replace(temporary, path)
    return tiles


class TileStore:
    """One bank per stock for the life of the process, so a frame never rebuilds a film."""

    def __init__(self, capacity: int = 3) -> None:
        self._entries: dict[str, StockTiles] = {}
        self._capacity = capacity

    def get(self, stock: GrainStock, levels: int = tb.TILE_LEVELS, seed: int = 1) -> StockTiles:
        signature = _signature(stock, levels, seed)
        key = f"{stock.stock_id}-{signature}"
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        tiles = load_or_build(stock, levels=levels, seed=seed)
        tiles.signature = signature
        if len(self._entries) >= self._capacity:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = tiles
        return tiles
