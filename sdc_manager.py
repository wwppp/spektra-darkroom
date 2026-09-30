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
            return True
        except Exception as e:
            logger.warning(f"Failed to delete SDC sidecar {sdc_path}: {e}")
            return False
    return False


def load_sdc(image_path: str) -> dict | None:
    """Load sidecar editing configuration if it exists.
    Returns dict with keys 'film_profile', 'paper_profile', 'params', or None if not found.
    """
    sdc_path = get_sdc_path(image_path)
    if not sdc_path or not os.path.isfile(sdc_path):
        return None

    try:
        with open(sdc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        logger.warning(f"Failed to read SDC sidecar {sdc_path}: {e}")
        return None


def save_sdc(image_path: str, film_profile: str, paper_profile: str, params: dict) -> bool:
    """Save sidecar editing configuration next to the image file."""
    sdc_path = get_sdc_path(image_path)
    if not sdc_path:
        return False

    try:
        data = {
            "format": "SpektraDarkroom Sidecar",
            "version": "1.0",
            "image_filename": os.path.basename(image_path),
            "film_profile": film_profile or "",
            "paper_profile": paper_profile or "",
            "params": params or {},
            "saved_at": datetime.now().isoformat()
        }
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
