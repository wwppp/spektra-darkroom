"""Which grain every stock carries, read from ``resources/profiles/grain_params.json``.

The table is baked from fotufilm packs by ``tools/sync_fotufilm_grain_params.py``; this module only
reads it.  A stock that is not in it gets **no grain** rather than somebody else's: a film whose
grain we have not measured is better off clean than wearing another stock's emulsion.

``source`` in the table is ``fotufilm:<pack>`` for a stock we took a same-name counterpart from,
and ``family:<family>`` where only a family default exists (P2 step three fills those in).  The UI
is meant to show the difference, so the value travels with the lookup.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .curve import Curve
from .population import CrystalGrainPopulation, GrainStock

__all__ = ["table", "stock_for", "available", "clear_cache"]

_LOCK = threading.Lock()
_CACHE: dict | None = None


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


def available() -> list[str]:
    return sorted(table().get("stocks", {}))


def stock_for(stock_id: str | None) -> GrainStock | None:
    """The grain a stock id carries, or ``None`` when its film has not been characterised."""
    if not stock_id:
        return None
    entry = table().get("stocks", {}).get(stock_id)
    if entry is None:
        return None
    curves = tuple(Curve.from_json(curve) for curve in entry["curves"])
    return GrainStock(
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


def source_for(stock_id: str | None) -> str | None:
    entry = table().get("stocks", {}).get(stock_id or "")
    return entry.get("source") if entry else None
