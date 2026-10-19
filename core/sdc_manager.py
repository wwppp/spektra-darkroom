"""Sidecar Configuration (.sdc) Manager for SpektraDarkroom
Saves and loads non-destructive editing parameters, film profiles, and paper profiles
alongside original image files.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("SpektraDarkroom.SDC")


def get_sdc_path(image_path: str) -> str:
    """Return the companion .sdc sidecar path for a given image file."""
    if not image_path:
        return ""
    return f"{image_path}.sdc"


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


import base64
import numpy as np
import cv2


def load_sdc(image_path: str) -> dict | None:
    """Load sidecar editing configuration if it exists.
    Returns dict with keys 'film_profile', 'paper_profile', 'params', and optional 'thumbnail_rgb'.
    """
    sdc_path = get_sdc_path(image_path)
    if not sdc_path or not os.path.isfile(sdc_path):
        return None

    try:
        with open(sdc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
            return data
    except Exception as e:
        logger.warning(f"Failed to read SDC sidecar {sdc_path}: {e}")
        return None


def save_sdc(image_path: str, film_profile: str, paper_profile: str, params: dict, thumbnail_rgb: np.ndarray = None, print_mode: str = "optical") -> bool:
    """Save sidecar editing configuration next to the image file, optionally caching thumbnail."""
    sdc_path = get_sdc_path(image_path)
    if not sdc_path:
        return False

    try:
        resolved_pm = print_mode or (params.get("print_mode", "optical") if params else "optical")
        data = {
            "format": "SpektraDarkroom Sidecar",
            "version": "1.0",
            "image_filename": os.path.basename(image_path),
            "film_profile": film_profile or "",
            "paper_profile": paper_profile or "",
            "print_mode": resolved_pm,
            "params": params or {},
            "saved_at": datetime.now().isoformat()
        }

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

        # Atomic write
        tmp_path = sdc_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if os.path.exists(sdc_path):
            os.remove(sdc_path)
        os.rename(tmp_path, sdc_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to write SDC sidecar {sdc_path}: {e}")
        return False
