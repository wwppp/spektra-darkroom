"""Which grain every stock carries, read from ``resources/profiles/grain_params.json``.

The table is baked from fotufilm packs by ``tools/sync_fotufilm_grain_params.py``; this module only
reads it.  A stock that is not in it gets **no grain** rather than somebody else's: a film whose
grain we have not measured is better off clean than wearing another stock's emulsion.

``source`` in the table is ``fotufilm:<pack>`` for a stock we took from that pack's sheet, and
``derived:<sibling>`` for a stock no pack covers: it borrows a relative's coating (population,
layer weights, crystal size) and carries its own strength, with ``derivation`` saying how.  The UI
is meant to show the difference, so the value travels with the lookup.
"""
from __future__ import annotations

import json
import pickle
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from .curve import Curve, SampledCurve
from .population import CrystalGrainPopulation, GrainStock

__all__ = ["table", "stock_for", "cached", "available", "clear_cache"]

_LOCK = threading.Lock()
# Separate from ``_LOCK``: ``stock_for`` reads the table (which takes ``_LOCK``) while holding
# this one, so sharing them would deadlock on the first cold build.
_BUILD_LOCK = threading.Lock()
_CACHE: dict | None = None
_STOCKS: dict[tuple[str, int], GrainStock] = {}


def _path() -> Path:
    from core.path_utils import get_resource_dir
    return Path(get_resource_dir()) / "profiles" / "grain_params.json"


def table() -> dict:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            path = _path()
            if not path.exists():
                _CACHE = {"version": 0, "stocks": {}}
            else:
                with open(path, "r", encoding="utf-8") as handle:
                    _CACHE = json.load(handle)
        return _CACHE


def clear_cache() -> None:
    """Drop the parsed table -- for the sync tool and for tests."""
    global _CACHE
    with _LOCK:
        _CACHE = None
        _STOCKS.clear()


def available() -> list[str]:
    return sorted(table().get("stocks", {}))


def cached(stock_id: str | None, push_stops: int = 0) -> GrainStock | None:
    """The already-built stock, or ``None``.  **Never builds.**

    Building a pushed stock runs ``refitDevelopment`` -- 9.5 s measured on kodak_gold_200 -- and
    the paint thread used to pay that inline, which is what made opening push look like a hang.
    The paint asks this instead, and a miss goes to a background build.
    """
    if not stock_id:
        return None
    return _STOCKS.get((stock_id, int(push_stops or 0)))


def _developed_cache_file(stock_id: str, stops: int) -> Path:
    """Where a pushed stock's built grain is kept between sessions.

    ``_pushed`` runs ``refitDevelopment`` -- 31 developabilities by four sublayers, 9.5 s measured
    on kodak_gold_200 -- and nothing about it depends on the session, only on the pack and the
    push table.  Recomputing it every launch is what made the grain take tens of seconds to
    appear.  The table's own mtime and size go into the name, so a re-synced pack misses.
    """
    from core.path_utils import get_user_cache_dir
    path = _path()
    try:
        stat = path.stat()
        table_tag = f"{stat.st_mtime_ns:x}{stat.st_size:x}"
    except OSError:
        table_tag = "0"
    return Path(get_user_cache_dir()) / "film_grain" / "stocks" / f"{stock_id}__{stops}__{table_tag}.pkl"


def _load_developed(stock_id: str, stops: int) -> GrainStock | None:
    path = _developed_cache_file(stock_id, stops)
    try:
        if not path.exists():
            return None
        import pickle as _p
        stock = _p.loads(path.read_bytes())
        return stock if isinstance(stock, GrainStock) else None
    except Exception:                                  # noqa: BLE001 - a bad cache is just a miss
        return None


def _save_developed(stock_id: str, stops: int, stock: GrainStock) -> None:
    try:
        path = _developed_cache_file(stock_id, stops)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(stock, protocol=4))
        tmp.replace(path)
    except Exception as exc:                           # noqa: BLE001 - caching must never fail a build
        print(f"[SpektraGrain] could not cache {stock_id} +{stops}: {exc}")


def stock_for(stock_id: str | None, push_stops: int = 0) -> GrainStock | None:
    """The grain a stock id carries, or ``None`` when its film has not been characterised.

    ``push_stops`` swaps in the over-developed characteristic curve (see ``_pushed``); zero is the
    stock's standard development and returns the table's own curves untouched.

    Memoised: the preview asks once a frame, and rebuilding three curves to answer the same
    question every 16 ms is work with no purpose.  Construction is serialised on ``_BUILD_LOCK``
    so two threads cannot both pay for the same 9.5 s refit.
    """
    if not stock_id:
        return None
    stops = int(push_stops or 0)
    key = (stock_id, stops)
    known = _STOCKS.get(key)
    if known is not None:
        return known
    with _BUILD_LOCK:
        known = _STOCKS.get(key)
        if known is not None:
            return known
        if stops:
            developed = _load_developed(stock_id, stops)
            if developed is not None:
                _STOCKS[key] = developed
                return developed
        entry = table().get("stocks", {}).get(stock_id)
        if entry is None:
            return None
        curves = tuple(Curve.from_json(curve) for curve in entry["curves"])
        stock = GrainStock(
            stock_id=stock_id,
            curves=curves,
            grain_size_mm=float(entry["grainSizeMM"]),
            grain_strength=float(entry["grainStrength"]),
            grain_layer_weights=tuple(float(w) for w in entry["grainLayerWeights"]),
            grain_layer_size_ratio=tuple(float(r) for r in entry.get("grainLayerSizeRatio", (1, 1, 1))),
            population=CrystalGrainPopulation.from_json(entry.get("crystalGrainPopulation")),
            is_monochrome=bool(entry.get("isMonochrome", False)),
            is_reversal=bool(entry.get("isReversal", False)),
            is_silver=bool(entry.get("isSilver", entry.get("isMonochrome", False))),
            grain_fog_density=float(entry.get("grainFogDensity", 0.03)),
            granularity_read_density=str(entry.get("granularityReadDensity", "net")),
            luma_correlation=float(entry.get("grainLumaCorrelation", 0.0)),
        )
        if stops:
            stock = _pushed(stock, stops)
            if stock.reference_id:            # a real refit happened; keep it
                _save_developed(stock_id, stops, stock)
        _STOCKS[key] = stock
        return stock


def _pushed(stock: GrainStock, stops: int) -> GrainStock:
    """``stock`` after ``stops`` of over-development; unchanged when it has no sidecar.

    The sidecar stores the condition as *net* density (``develop()`` baseline-zeroes whatever it
    is handed, so fog cannot live in the curves) plus ``dMin``, the fog the extra development adds.
    Rebuilding gross density is therefore

        gross = net + <standard curve>.d_min + dMin

    and that is the whole point of the exercise: the grain law is indexed by net density, and a
    pushed negative reaches the same net density from a *thicker* piece of film, so the same
    exposure reads a coarser crystal class.

    Only the records move.  Crystal size, strength, layer weights and the population stay where
    the pack measured them -- this function is "the same emulsion developed longer", not a
    different stock, and inventing a per-stop grain multiplier is exactly what ``CHANGELOG``
    2026-11-02 removed.
    """
    from core import push_model
    from core.path_utils import get_resource_dir

    table = push_model.load_table(str(get_resource_dir()), stock.stock_id)
    entry, applied = push_model.condition(table, stops)
    if entry is None or not applied:
        return stock
    log_exposure = np.asarray(table["log_exposure"], dtype=np.float64)
    net = np.asarray(entry["curves"], dtype=np.float64)
    if net.shape[0] != log_exposure.size or net.shape[1] != len(stock.curves):
        return stock
    fog = float(entry.get("dMin") or 0.0)
    records = []
    for record, base in enumerate(stock.curves):
        gross = net[:, record] + float(base.d_min) + fog
        records.append(replace(
            base,
            d_min=float(np.nanmin(gross)),
            sampled=SampledCurve.from_points(log_exposure, gross),
        ))
    pushed = replace(stock, curves=tuple(records), reference_id=stock.stock_id)
    return replace(pushed, anchor_scales=_anchor_scales(stock, pushed))


def _anchor_scales(base: GrainStock, pushed: GrainStock) -> tuple[float, float, float]:
    """σ_model(developed) / σ_model(standard) at the read density, per record.

    The published granularity ``read_sigma`` was measured at the *reference* process.  A pushed
    negative reaches the same net density from a thicker piece of film, so the model -- now that
    ``refitDevelopment`` has solved the dye gain -- predicts a coarser grain there.  Anchoring
    straight back to ``read_sigma`` would erase exactly the difference the push is supposed to
    have, so the anchor target becomes ``read_sigma * scale``.

    This is a deliberate divergence from the reference engine, which has no condition-level
    granularity and therefore cannot express it (``docs/push-port-plan.md`` T1-e).  Standard
    development never calls this, so its scale is 1.0 by construction.
    """
    from .population import CrystalPopulation

    scales: list[float] = []
    for record in range(len(base.curves)):
        try:
            standard = CrystalPopulation(base, record)
            developed = CrystalPopulation(pushed, record, reference=base)
            at = float(standard.read_net_density)
            reference = standard.sigma(at)
            scales.append(float(developed.sigma(at) / reference) if reference > 1e-9 else 1.0)
        except Exception:                            # noqa: BLE001 - never fail a stock over this
            scales.append(1.0)
    while len(scales) < 3:
        scales.append(scales[-1] if scales else 1.0)
    return tuple(scales[:3])


def source_for(stock_id: str | None) -> str | None:
    entry = table().get("stocks", {}).get(stock_id or "")
    return entry.get("source") if entry else None
