"""Sidecar Configuration (.sdc) Manager for SpektraDarkroom
Saves and loads non-destructive editing parameters, film profiles, paper profiles,
and cached thumbnails alongside original image files.
Star ratings belong exclusively to project session files (.sdss).
"""

import os
import json
import logging
from datetime import datetime
import base64
import numpy as np
import cv2

logger = logging.getLogger("SpektraDarkroom.SDC")


def get_sdc_path(image_path: str) -> str:
    """Return the companion .sdc sidecar path for a given image file."""
    if not image_path:
        return ""
    return f"{image_path}.sdc"


# The two Portra 800 push stocks used to be full profiles, i.e. pushing was expressed by picking a
# different film from the list.  Push is a development condition now (one profile + a table), so
# those ids are retired: a sidecar written back then means "Portra 800, developed +1/+2".
# The profile files stay on disk (old sessions must still load) and stay in BUILTIN_FILM_IDS so
# they are never mistaken for user-deleted customs -- they are only unreachable from the film list.
LEGACY_PUSH_PROFILES = {
    "kodak_portra_800_push1": ("kodak_portra_800", 1),
    "kodak_portra_800_push2": ("kodak_portra_800", 2),
}


def migrate_legacy_push(data: dict) -> dict:
    """Rewrite a retired push stock id as its base stock plus ``push_ev``.

    Idempotent: a sidecar that already names the base stock is left untouched.  Applying it on
    *load* (not on save) is what makes it free -- the next save writes the new form.
    """
    if not isinstance(data, dict):
        return data
    film = data.get("film_profile")
    if film in LEGACY_PUSH_PROFILES:
        base, stops = LEGACY_PUSH_PROFILES[film]
        data["film_profile"] = base
        params = data.get("params")
        if isinstance(params, dict):
            params["push_ev"] = stops
        logger.info(f"Migrated legacy push stock {film} -> {base} push_ev={stops}")
    return data


def migrate_push_coupling(data: dict) -> dict:
    """Give a pre-flag sidecar its old meaning: push used to drag the exposure down by itself.

    ``push_underexpose`` is what makes that linkage optional.  A sidecar written before it
    existed was already rendered with the underexposure baked into ``exposure_ev``, so restoring
    it as "unlinked" would make the next +N click subtract the stops a second time.
    """
    params = data.get("params") if isinstance(data, dict) else None
    if not isinstance(params, dict):
        return data
    if "push_underexpose" not in params and int(params.get("push_ev", 0) or 0):
        params["push_underexpose"] = True
    return data


def has_sdc(image_path: str) -> bool:
    sdc_path = get_sdc_path(image_path)
    return bool(sdc_path and os.path.isfile(sdc_path))


def delete_sdc(image_path: str) -> bool:
    """Delete companion .sdc sidecar file if present."""
    sdc_path = get_sdc_path(image_path)
    if sdc_path and os.path.isfile(sdc_path):
        try:
            os.remove(sdc_path)
            try:
                from . import session_cache_manager
                session_cache_manager.invalidate_photo_cache(image_path)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.warning(f"Failed to delete SDC sidecar {sdc_path}: {e}")
            return False
    return False


# Backward compatibility alias
remove_sdc = delete_sdc


def load_sdc(image_path: str) -> dict | None:
    """Load sidecar editing configuration if it exists.
    Returns dict with keys 'film_profile', 'paper_profile', 'params', and optional 'thumbnail_rgb'.
    Note: Ratings belong exclusively to session files (.sdss) and are never loaded from .sdc sidecars.
    """
    sdc_path = get_sdc_path(image_path)
    if not sdc_path or not os.path.isfile(sdc_path):
        return None

    try:
        with open(sdc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Guarantee rating is never loaded from SDC
            data.pop("rating", None)
            # Restore cached thumbnail if present
            if "thumbnail_b64" in data:
                try:
                    raw_bytes = base64.b64decode(data["thumbnail_b64"])
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        data["thumbnail_rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
            # Type normalisation: consumers call sdc["params"].get(...) directly, so a
            # null or a string here would raise AttributeError deep in the UI.
            if not isinstance(data.get("params"), dict):
                data["params"] = {}
            # 旁挂的另一条链的相纸选择：类型在这里就规范好，消费方直接 .get()。
            by_scope = data.get("paper_by_scope")
            if not isinstance(by_scope, dict):
                by_scope = {}
            data["paper_by_scope"] = {
                str(k): str(v) for k, v in by_scope.items()
                if k in ("native", "ffchain") and v}
            # base_ev is the auto-metering anchor for the *file*, not a user edit. It is
            # derived from the decoded preview and cached against the file's mtime+size.
            # Older sidecars carry a copy; honouring it made the rendered exposure depend on
            # visit history (session entry / sidecar / cache could disagree by stops), which
            # showed up as "the photo got brighter after I switched away and back".
            data["params"].pop("base_ev", None)
            return migrate_push_coupling(migrate_legacy_push(data))
    except Exception as e:
        logger.warning(f"Failed to read SDC sidecar {sdc_path}: {e}")
        # Quarantine instead of silently pretending the photo was never edited: the next
        # autosave would otherwise overwrite the user's recipe with defaults.
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            corrupt_path = f"{sdc_path}.corrupt-{stamp}"
            os.replace(sdc_path, corrupt_path)
            logger.warning(f"Corrupted SDC sidecar moved to {corrupt_path}")
        except Exception as move_err:
            logger.warning(f"Could not quarantine corrupted SDC sidecar {sdc_path}: {move_err}")
        return None


def save_sdc(
    image_path: str,
    film_profile: str,
    paper_profile: str,
    params: dict,
    thumbnail_rgb: np.ndarray = None,
    print_mode: str = None,
    rating: int = None,
    paper_explicit: bool = None,
    paper_by_scope: dict = None
) -> bool:
    """Save sidecar editing configuration next to the image file, optionally caching thumbnail.
    Note: Star ratings belong exclusively to session files (.sdss) and are never written to .sdc sidecars.
    """
    sdc_path = get_sdc_path(image_path)
    if not sdc_path:
        return False

    try:
        resolved_pm = print_mode
        existing = load_sdc(image_path)
        if not resolved_pm and params and isinstance(params, dict):
            resolved_pm = params.get("print_mode")
        if not resolved_pm and existing and existing.get("print_mode"):
            resolved_pm = existing.get("print_mode")
        if not resolved_pm:
            resolved_pm = "optical"

        data = {
            "format": "SpektraDarkroom Sidecar",
            "version": "1.0",
            "image_filename": os.path.basename(image_path),
            "film_profile": film_profile or "",
            "paper_profile": paper_profile or "",
            "print_mode": resolved_pm,
            # The metering anchor is stripped on the way out as well: it is re-derived from
            # the file (see load_sdc) and must never be restored from a sidecar.
            "params": {k: v for k, v in (params or {}).items() if k != "base_ev"},
            "saved_at": datetime.now().isoformat()
        }

        # 相纸是用户选的还是自动补的默认值。空 paper_profile 不区分这两者，而首选项换默认
        # 相纸时只有"没选过"的照片才该跟着变，所以这个标记必须落到盘上。
        if paper_explicit is not None:
            data["paper_explicit"] = bool(paper_explicit)

        # 另一条链上的相纸选择（`{"native": id, "ffchain": id}`）。`paper_profile` 永远存
        # 当前卷所属那条链的合法值，这份只是旁挂 —— 两条链的纸不相通，只用一个字段时
        # 切卷会把用户在另一条链上选过的纸覆写掉，切回去也回不来。
        if paper_by_scope is not None:
            clean = {str(k): str(v) for k, v in (paper_by_scope or {}).items()
                     if k in ("native", "ffchain") and v}
            if clean:
                data["paper_by_scope"] = clean

        # Embed compact JPEG thumbnail (~2KB) so any machine/session opens with instant graded preview
        if thumbnail_rgb is not None and isinstance(thumbnail_rgb, np.ndarray) and thumbnail_rgb.size > 0:
            try:
                th_u8 = thumbnail_rgb
                if th_u8.dtype != np.uint8:
                    th_u8 = (np.clip(th_u8, 0.0, 1.0) * 255.0).astype(np.uint8)
                th_bgr = cv2.cvtColor(th_u8, cv2.COLOR_RGB2BGR)
                success, enc = cv2.imencode(".jpg", th_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if success:
                    data["thumbnail_b64"] = base64.b64encode(enc.tobytes()).decode("ascii")
            except Exception as ex:
                logger.debug(f"Failed encoding thumbnail for SDC {sdc_path}: {ex}")

        # Carry over the previously embedded thumbnail when this save has none, so that
        # routine parameter saves do not silently drop the offline preview.
        if "thumbnail_b64" not in data and os.path.isfile(sdc_path):
            try:
                with open(sdc_path, "r", encoding="utf-8") as prev_f:
                    prev_data = json.load(prev_f)
                if isinstance(prev_data, dict) and prev_data.get("thumbnail_b64"):
                    data["thumbnail_b64"] = prev_data["thumbnail_b64"]
            except Exception:
                pass

        # Atomic write with os.replace: the payload lands in .tmp first, so a crash
        # mid-write can never truncate the sidecar. A .bak copy used to be kept here on
        # every single save, which meant one junk file per edit next to every photo --
        # the .tmp + replace pair already gives the rollback guarantee that mattered.
        tmp_path = sdc_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, sdc_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to write SDC sidecar {sdc_path}: {e}")
        return False


def update_sdc_rating(image_path: str, rating: int) -> bool:
    """Deprecated: Star ratings belong exclusively to session files (.sdss).
    Kept as a safe no-op for backward compatibility.
    """
    return True
