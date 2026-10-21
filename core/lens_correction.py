"""lens_correction.py - Lens Optical & Geometric Correction Engine for SpektraDarkroom
Supports:
1. Adobe Camera Raw .lcp XML profile parsing (RadialDistortCoeff & VignetteModelCoeff)
2. RAW / DNG / EXIF metadata extraction for automatic lens identification
3. Unified radial polynomial calculation for GPU GLSL Fragment Shader and CPU fallback
"""

import os
import re
import math
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("SpektraDarkroom.Lens")


@dataclass
class LensCalibrationEntry:
    focal_length: float = 50.0
    aperture: float = 2.8
    focus_distance: float = 3.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    vignette_coeffs: List[float] = field(default_factory=list)


@dataclass
class LensProfile:
    maker: str = ""
    model: str = ""
    lens_name: str = ""
    sensor_format_factor: float = 1.0  # Crop factor (1.0 = Full Frame, 1.5 = APS-C)
    calibrations: List[LensCalibrationEntry] = field(default_factory=list)
    source_file: str = ""

    def get_best_calibration(self, focal_length: Optional[float] = None, aperture: Optional[float] = None) -> LensCalibrationEntry:
        """Find the calibration entry nearest to the given focal length and aperture."""
        if not self.calibrations:
            return LensCalibrationEntry()

        fl = focal_length if focal_length is not None and focal_length > 0 else 50.0
        ap = aperture if aperture is not None and aperture > 0 else 2.8

        best_entry = self.calibrations[0]
        best_dist = float("inf")

        for entry in self.calibrations:
            # Weighted distance: focal length is much more critical for distortion than aperture
            fl_diff = abs(entry.focal_length - fl)
            ap_diff = abs(entry.aperture - ap)
            dist = fl_diff * 10.0 + ap_diff
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        return best_entry


class LCPParser:
    """Parser for Adobe Camera Raw .lcp (Lens Correction Profile) XML files."""

    NS = {
        "x": "adobe:ns:meta/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "stCamera": "http://ns.adobe.com/camera-profile/1.0/"
    }

    @classmethod
    def parse_file(cls, lcp_path: str) -> Optional[LensProfile]:
        if not lcp_path or not os.path.isfile(lcp_path):
            return None

        try:
            tree = ET.parse(lcp_path)
            root = tree.getroot()

            # Find all elements regardless of exact namespace prefix matching
            lens_name = ""
            maker = ""
            model = ""
            crop_factor = 1.0
            calibrations = []

            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag in ("LensPrettyName", "Lens") and not lens_name:
                    lens_name = (elem.text or "").strip()
                elif tag in ("CameraPrettyName", "Model") and not model:
                    model = (elem.text or "").strip()
                elif tag == "Author" and not maker:
                    maker = (elem.text or "").strip()
                elif tag == "SensorFormatFactor":
                    try:
                        crop_factor = float(elem.text.strip())
                    except Exception:
                        pass

            # Search PerspectiveModel / Seq / li items for calibrations
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "li":
                    fl = 50.0
                    ap = 2.8
                    distort_str = ""
                    vig_str = ""

                    for child in elem:
                        ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                        if ctag == "FocalLength":
                            try:
                                fl = float(child.text.strip())
                            except Exception:
                                pass
                        elif ctag == "ApertureValue":
                            try:
                                ap = float(child.text.strip())
                            except Exception:
                                pass
                        elif ctag == "RadialDistortCoeff":
                            distort_str = (child.text or "").strip()
                        elif ctag == "VignetteModelCoeff":
                            vig_str = (child.text or "").strip()

                    if distort_str or vig_str:
                        k1, k2, k3 = 0.0, 0.0, 0.0
                        if distort_str:
                            parts = [float(p) for p in re.split(r"[\s,]+", distort_str) if p]
                            if len(parts) >= 1:
                                k1 = parts[0]
                            if len(parts) >= 2:
                                k2 = parts[1]
                            if len(parts) >= 3:
                                k3 = parts[2]

                        vig_coeffs = []
                        if vig_str:
                            vig_coeffs = [float(p) for p in re.split(r"[\s,]+", vig_str) if p]

                        calibrations.append(LensCalibrationEntry(
                            focal_length=fl,
                            aperture=ap,
                            k1=k1,
                            k2=k2,
                            k3=k3,
                            vignette_coeffs=vig_coeffs
                        ))

            if not lens_name and calibrations:
                lens_name = os.path.splitext(os.path.basename(lcp_path))[0]

            if not lens_name:
                return None

            return LensProfile(
                maker=maker,
                model=model,
                lens_name=lens_name,
                sensor_format_factor=crop_factor,
                calibrations=calibrations,
                source_file=lcp_path
            )
        except Exception as e:
            logger.warning(f"Failed to parse LCP file {lcp_path}: {e}")
            return None


def read_lens_metadata(file_path: str) -> Dict[str, any]:
    """Extract lens, camera, focal length, aperture, and DNG opcodes from RAW or standard image file."""
    meta = {
        "camera_make": "",
        "camera_model": "",
        "lens_model": "",
        "focal_length": None,
        "aperture": None,
        "iso": None,
        "has_embedded_correction": False,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "vignette_amount": 0.0
    }

    if not file_path or not os.path.isfile(file_path):
        return meta

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Try rawpy for RAW formats
    raw_extensions = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef"}
    if ext in raw_extensions:
        try:
            import rawpy
            with rawpy.imread(file_path) as r:
                if hasattr(r, "lens"):
                    lens_obj = r.lens
                    meta["lens_model"] = getattr(lens_obj, "model", "") or ""
                if hasattr(r, "other"):
                    other_obj = r.other
                    meta["focal_length"] = getattr(other_obj, "focal_len", None)
                    meta["aperture"] = getattr(other_obj, "aperture", None)
                    meta["iso"] = getattr(other_obj, "iso_speed", None)
        except Exception as e:
            logger.debug(f"rawpy metadata read error for {file_path}: {e}")

    # 2. Try PIL EXIF for all formats
    try:
        from PIL import Image, ExifTags
        with Image.open(file_path) as img:
            exif = img.getexif()
            if exif:
                # 271: Make, 272: Model
                if not meta["camera_make"] and 271 in exif:
                    meta["camera_make"] = str(exif[271]).strip()
                if not meta["camera_model"] and 272 in exif:
                    meta["camera_model"] = str(exif[272]).strip()

                # Search ExifIFD (tag 34665)
                exif_ifd = exif.get_ifd(34665) if hasattr(exif, "get_ifd") else None
                if exif_ifd:
                    # 42036: LensModel, 42035: LensMake
                    if not meta["lens_model"] and 42036 in exif_ifd:
                        meta["lens_model"] = str(exif_ifd[42036]).strip()
                    # 37386: FocalLength
                    if meta["focal_length"] is None and 37386 in exif_ifd:
                        try:
                            fl_val = exif_ifd[37386]
                            meta["focal_length"] = float(fl_val[0]) / float(fl_val[1]) if isinstance(fl_val, tuple) else float(fl_val)
                        except Exception:
                            pass
                    # 33437: FNumber
                    if meta["aperture"] is None and 33437 in exif_ifd:
                        try:
                            fn_val = exif_ifd[33437]
                            meta["aperture"] = float(fn_val[0]) / float(fn_val[1]) if isinstance(fn_val, tuple) else float(fn_val)
                        except Exception:
                            pass
    except Exception as e:
        logger.debug(f"PIL EXIF read error for {file_path}: {e}")

    # Fallback lens description if lens model is empty but camera model is present
    if not meta["lens_model"] and meta["camera_model"]:
        fl_str = f"{int(meta['focal_length'])}mm" if meta['focal_length'] else ""
        meta["lens_model"] = f"{meta['camera_model']} {fl_str}".strip()

    return meta


def find_system_adobe_lcp_directories() -> List[str]:
    """Scan standard Windows installation paths for Adobe Camera Raw LensProfiles."""
    candidates = [
        r"C:\ProgramData\Adobe\CameraRaw\LensProfiles\1.0",
        r"C:\Program Files\Common Files\Adobe\CameraRaw\LensProfiles\1.0",
        r"C:\Program Files\Adobe\Adobe Lightroom Classic\Resources\LensProfiles\1.0",
        os.path.expandvars(r"%APPDATA%\Adobe\CameraRaw\LensProfiles\1.0")
    ]
    return [p for p in candidates if os.path.isdir(p)]


def compute_manual_distortion_coeffs(distortion_value: float) -> Tuple[float, float, float]:
    """Convert slider value (-100 to +100) into radial polynomial coefficients (k1, k2, k3).
    Negative = Pincushion correction (compensate barrel, pull in corners)
    Positive = Barrel correction (compensate pincushion, push out corners)
    """
    if abs(distortion_value) < 0.001:
        return 0.0, 0.0, 0.0

    # Non-linear mapping: 100 maps to ~0.20 coefficient
    val = float(distortion_value) / 100.0
    k1 = val * 0.22
    k2 = (val ** 2) * 0.06 if val > 0 else -(val ** 2) * 0.06
    k3 = 0.0
    return k1, k2, k3


def compute_manual_vignette_coeffs(amount: float, midpoint: float) -> Tuple[float, float]:
    """Convert amount (0 to 200%) and midpoint (10 to 90) into shader vignette parameters."""
    if amount < 0.001:
        return 0.0, 50.0
    return float(amount) / 100.0, float(midpoint)
