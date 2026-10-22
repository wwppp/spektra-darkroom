"""lens_correction.py - Lens Optical & Geometric Correction Engine for SpektraDarkroom
Supports:
1. Camera Raw .lcp XML profile parsing (attributes & child elements)
2. RAW / DNG / EXIF metadata extraction for automatic lens identification
3. Unified radial polynomial calculation for GPU GLSL Fragment Shader and CPU fallback
4. System-wide LensProfiles indexer with brand/model hierarchy
"""

import os
import re
import math
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("SpektraDarkroom.Lens")


def _parse_float_safe(val: any) -> Optional[float]:
    """Safely parse positive float from float, int, or display strings like '17 毫米', 'f/8.0', '50mm'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if float(val) > 0 else None
    s = str(val).strip()
    if not s or s == "-":
        return None
    m = re.search(r"\d+\.?\d*", s)
    if m:
        try:
            f = float(m.group(0))
            return f if f > 0 else None
        except ValueError:
            return None
    return None


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
    profile_name: str = ""
    sensor_format_factor: float = 1.0  # Crop factor (1.0 = Full Frame, 1.5 = APS-C)
    calibrations: List[LensCalibrationEntry] = field(default_factory=list)
    source_file: str = ""

    def get_best_calibration(self, focal_length: Optional[float] = None, aperture: Optional[float] = None) -> LensCalibrationEntry:
        """Find the calibration entry nearest to the given focal length and aperture."""
        if not self.calibrations:
            return LensCalibrationEntry()

        fl = _parse_float_safe(focal_length)
        if fl is None:
            # If no focal length provided in image EXIF, pick calibration at wide end of lens
            min_fl_entry = min(self.calibrations, key=lambda c: c.focal_length)
            fl = min_fl_entry.focal_length

        ap_parsed = _parse_float_safe(aperture)
        ap = ap_parsed if ap_parsed is not None else 2.8

        best_entry = self.calibrations[0]
        best_dist = float("inf")

        for entry in self.calibrations:
            fl_diff = abs(entry.focal_length - fl)
            ap_diff = abs(entry.aperture - ap)
            dist = fl_diff * 10.0 + ap_diff
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        return best_entry


class LCPParser:
    """Robust parser for Camera Raw .lcp (Lens Correction Profile) XML files."""

    @classmethod
    def parse_file(cls, lcp_path: str) -> Optional[LensProfile]:
        if not lcp_path or not os.path.isfile(lcp_path):
            return None

        try:
            tree = ET.parse(lcp_path)
            root = tree.getroot()

            lens_name = ""
            maker = ""
            model = ""
            profile_name = ""
            crop_factor = 1.0
            calibrations = []

            # 1. Global metadata pass
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                attr = {k.split("}")[-1]: v for k, v in elem.attrib.items()}

                if not lens_name:
                    lens_name = attr.get("LensPrettyName") or attr.get("Lens") or (elem.text if tag in ("LensPrettyName", "Lens") else "")
                if not maker:
                    maker = attr.get("Make") or attr.get("CameraPrettyName") or (elem.text if tag in ("Make", "Author", "CameraPrettyName") else "")
                if not model:
                    model = attr.get("Model") or (elem.text if tag in ("Model",) else "")
                if not profile_name:
                    profile_name = attr.get("ProfileName") or (elem.text if tag == "ProfileName" else "")

                if "SensorFormatFactor" in attr:
                    try:
                        crop_factor = float(attr["SensorFormatFactor"])
                    except Exception:
                        pass
                elif tag == "SensorFormatFactor" and elem.text:
                    try:
                        crop_factor = float(elem.text.strip())
                    except Exception:
                        pass

            # 2. Hierarchical calibration entries pass
            for desc in root.iter():
                tag = desc.tag.split("}")[-1] if "}" in desc.tag else desc.tag
                if tag != "Description":
                    continue

                desc_attr = {k.split("}")[-1]: v for k, v in desc.attrib.items()}
                has_fl = "FocalLength" in desc_attr
                has_model_child = any(c.tag.split("}")[-1] in ("PerspectiveModel", "FisheyeModel", "VignetteModel") for c in desc)
                if not has_fl and not has_model_child:
                    continue

                fl = None
                if "FocalLength" in desc_attr:
                    try:
                        fl = float(desc_attr["FocalLength"])
                    except Exception:
                        pass

                ap = None
                if "ApertureValue" in desc_attr:
                    try:
                        raw_ap = float(desc_attr["ApertureValue"])
                        # Adobe ApertureValue is APEX: F# = 2 ^ (Av / 2)
                        ap = (2.0 ** (raw_ap / 2.0)) if raw_ap < 15.0 else raw_ap
                    except Exception:
                        pass

                fd = 3.0
                if "FocusDistance" in desc_attr:
                    try:
                        fd = float(desc_attr["FocusDistance"])
                    except Exception:
                        pass

                k1, k2, k3 = 0.0, 0.0, 0.0
                vig_coeffs = []

                for sub in desc.iter():
                    sub_attr = {k.split("}")[-1]: v for k, v in sub.attrib.items()}

                    if "RadialDistortParam1" in sub_attr:
                        try:
                            k1 = float(sub_attr["RadialDistortParam1"])
                            k2 = float(sub_attr.get("RadialDistortParam2", 0.0))
                            k3 = float(sub_attr.get("RadialDistortParam3", 0.0))
                        except Exception:
                            pass
                    elif "RadialDistortCoeff" in sub_attr:
                        try:
                            parts = [float(p) for p in re.split(r"[\s,]+", sub_attr["RadialDistortCoeff"].strip()) if p]
                            if len(parts) >= 1: k1 = parts[0]
                            if len(parts) >= 2: k2 = parts[1]
                            if len(parts) >= 3: k3 = parts[2]
                        except Exception:
                            pass

                    if "VignetteModelParam1" in sub_attr:
                        try:
                            vig_coeffs = [
                                float(sub_attr["VignetteModelParam1"]),
                                float(sub_attr.get("VignetteModelParam2", 0.0)),
                                float(sub_attr.get("VignetteModelParam3", 0.0))
                            ]
                        except Exception:
                            pass
                    elif "VignetteModelCoeff" in sub_attr:
                        try:
                            vig_coeffs = [float(p) for p in re.split(r"[\s,]+", sub_attr["VignetteModelCoeff"].strip()) if p]
                        except Exception:
                            pass

                if abs(k1) > 1e-6 or abs(k2) > 1e-6 or abs(k3) > 1e-6 or vig_coeffs:
                    calibrations.append(LensCalibrationEntry(
                        focal_length=fl if fl is not None else 50.0,
                        aperture=ap if ap is not None else 2.8,
                        focus_distance=fd,
                        k1=k1,
                        k2=k2,
                        k3=k3,
                        vignette_coeffs=vig_coeffs
                    ))

            if not lens_name:
                lens_name = os.path.splitext(os.path.basename(lcp_path))[0]
            if not profile_name:
                profile_name = lens_name

            return LensProfile(
                maker=maker.strip(),
                model=model.strip(),
                lens_name=lens_name.strip(),
                profile_name=profile_name.strip(),
                sensor_format_factor=crop_factor,
                calibrations=calibrations,
                source_file=lcp_path
            )
        except Exception as e:
            logger.warning(f"Failed to parse LCP file {lcp_path}: {e}")
            return None


def find_system_lcp_directories() -> List[str]:
    """Scan standard system installation paths for Camera Raw LensProfiles (.lcp)."""
    candidates = [
        r"C:\ProgramData\Adobe\CameraRaw\LensProfiles\1.0",
        r"C:\Program Files\Common Files\Adobe\CameraRaw\LensProfiles\1.0",
        r"C:\Program Files\Adobe\Adobe Lightroom Classic\Resources\LensProfiles\1.0",
        os.path.expandvars(r"%APPDATA%\Adobe\CameraRaw\LensProfiles\1.0")
    ]
    return [p for p in candidates if os.path.isdir(p)]

# Backward compatibility alias
find_system_adobe_lcp_directories = find_system_lcp_directories


class LCPDatabase:
    """In-memory indexer and manager for Lens Profiles (.lcp)."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "LCPDatabase":
        if cls._instance is None:
            cls._instance = LCPDatabase()
        return cls._instance

    def __init__(self):
        self._roots: List[str] = find_system_lcp_directories()
        self._makes: List[str] = []
        self._models_by_make: Dict[str, List[Tuple[str, str, str]]] = {}  # make -> [(model_name, profile_name, path)]
        self._profile_cache: Dict[str, LensProfile] = {}
        self._indexed = False
        self._build_index()

    def index_installed_profiles(self):
        """Build or refresh profile database index."""
        self._build_index()

    def _build_index(self):
        if not self._roots:
            return

        makes_set = set()
        for root_dir in self._roots:
            try:
                for entry in os.listdir(root_dir):
                    sub = os.path.join(root_dir, entry)
                    if os.path.isdir(sub):
                        makes_set.add(entry)
            except Exception:
                pass

        self._makes = sorted(list(makes_set), key=lambda s: s.lower())
        self._indexed = True

    def get_available_makes(self) -> List[str]:
        return list(self._makes)

    def get_models_for_make(self, make: str) -> List[Tuple[str, str, str]]:
        """Return list of (model_display_name, profile_display_name, file_path) for the given make."""
        if not make:
            return []
        if make in self._models_by_make:
            return self._models_by_make[make]

        results = []
        for root_dir in self._roots:
            make_dir = os.path.join(root_dir, make)
            if not os.path.isdir(make_dir):
                continue
            for r, dirs, files in os.walk(make_dir):
                for f in files:
                    if f.lower().endswith(".lcp"):
                        p = os.path.join(r, f)
                        # Derive model name cleanly from filename or quick metadata
                        clean_name = f[:-4]
                        # Remove leading brand prefix if redundant
                        if clean_name.upper().startswith(make.upper()):
                            clean_name = clean_name[len(make):].strip(" -_()")
                        if clean_name.endswith(" - RAW") or clean_name.endswith(" - JPEG"):
                            clean_name = clean_name[:-7].strip()
                        if not clean_name:
                            clean_name = f[:-4]
                        prof_display = clean_name
                        results.append((clean_name, prof_display, p))

        # Sort by model name
        results.sort(key=lambda x: x[0].lower())
        self._models_by_make[make] = results
        return results

    def load_profile(self, lcp_path: str) -> Optional[LensProfile]:
        if not lcp_path or not os.path.isfile(lcp_path):
            return None
        if lcp_path in self._profile_cache:
            return self._profile_cache[lcp_path]

        prof = LCPParser.parse_file(lcp_path)
        if prof:
            self._profile_cache[lcp_path] = prof
        return prof

    def match_lens_from_exif(self, camera_make: str, camera_model: str, lens_model: str, focal_length: Optional[float] = None) -> Optional[Tuple[str, str, str, str]]:
        """Attempt to find matching (make, model, profile_name, lcp_path) from EXIF metadata."""
        if not self._makes:
            return None

        # 1. Determine make candidates
        make_candidates = []
        clean_lens = (lens_model or "").strip()
        clean_cam_make = (camera_make or "").strip()

        # Check if lens_model starts with or contains any known make
        for m in self._makes:
            if m.lower() in clean_lens.lower():
                make_candidates.append(m)
        if clean_cam_make:
            for m in self._makes:
                if m.lower() == clean_cam_make.lower() and m not in make_candidates:
                    make_candidates.append(m)

        if not make_candidates and self._makes:
            make_candidates = list(self._makes)

        # 2. Search inside candidate makes
        lens_tokens = set(re.findall(r"[A-Za-z0-9]+", clean_lens.lower()))
        best_match = None
        best_score = 0

        for m in make_candidates:
            models = self.get_models_for_make(m)
            for mod_name, prof_name, p in models:
                mod_tokens = set(re.findall(r"[A-Za-z0-9]+", mod_name.lower()))
                common = lens_tokens.intersection(mod_tokens)
                score = len(common)
                if score > best_score and score >= 2:
                    best_score = score
                    best_match = (m, mod_name, prof_name, p)

        return best_match


def read_lens_metadata(file_path: str) -> Dict[str, any]:
    """Extract lens, camera, focal length, aperture, and metadata from RAW or standard image file."""
    meta = {
        "camera_make": "",
        "camera_model": "",
        "lens_model": "",
        "focal_length": None,
        "aperture": None,
        "iso": None
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
                if not meta["camera_make"] and 271 in exif:
                    meta["camera_make"] = str(exif[271]).strip()
                if not meta["camera_model"] and 272 in exif:
                    meta["camera_model"] = str(exif[272]).strip()

                exif_ifd = exif.get_ifd(34665) if hasattr(exif, "get_ifd") else None
                if exif_ifd:
                    if not meta["lens_model"] and 42036 in exif_ifd:
                        meta["lens_model"] = str(exif_ifd[42036]).strip()
                    if meta["focal_length"] is None and 37386 in exif_ifd:
                        try:
                            fl_val = exif_ifd[37386]
                            meta["focal_length"] = float(fl_val[0]) / float(fl_val[1]) if isinstance(fl_val, tuple) else float(fl_val)
                        except Exception:
                            pass
                    if meta["aperture"] is None and 33437 in exif_ifd:
                        try:
                            fn_val = exif_ifd[33437]
                            meta["aperture"] = float(fn_val[0]) / float(fn_val[1]) if isinstance(fn_val, tuple) else float(fn_val)
                        except Exception:
                            pass
    except Exception as e:
        logger.debug(f"PIL EXIF read error for {file_path}: {e}")

    meta["focal_length"] = _parse_float_safe(meta.get("focal_length"))
    meta["aperture"] = _parse_float_safe(meta.get("aperture"))

    if not meta["lens_model"] and meta["camera_model"]:
        fl_str = f"{int(meta['focal_length'])}mm" if meta['focal_length'] else ""
        meta["lens_model"] = f"{meta['camera_model']} {fl_str}".strip()

    return meta


def compute_manual_distortion_coeffs(distortion_value: float) -> Tuple[float, float, float]:
    """Convert slider value (-100 to +100) into radial polynomial coefficients (k1, k2, k3).
    Negative = Pincushion correction (compensate barrel, pull in corners)
    Positive = Barrel correction (compensate pincushion, push out corners)
    """
    if abs(distortion_value) < 0.001:
        return 0.0, 0.0, 0.0

    val = float(distortion_value) / 100.0
    k1 = val * 0.15
    k2 = (val ** 2) * 0.04 if val > 0 else -(val ** 2) * 0.04
    k3 = 0.0
    return k1, k2, k3


def compute_manual_vignette_coeffs(amount: float, midpoint: float) -> Tuple[float, float]:
    """Convert amount (-100 to +100%) and midpoint (-100 to +100) into shader vignette parameters."""
    if abs(amount) < 0.001:
        v_amt = 0.0
    elif amount > 0:
        v_amt = float(amount) / 100.0  # 0.0 to +1.0 (brighten corners)
    else:
        v_amt = (float(amount) / 100.0) * 0.9  # -0.9 to 0.0 (darkens corners)

    # Midpoint slider 0 -> 50.0; -100 -> 10.0; +100 -> 90.0
    v_mid = 50.0 + (float(midpoint) * 0.4)
    return v_amt, v_mid


def compute_effective_lens_params(lens_cfg: dict, focal_length: Optional[float] = None, aperture: Optional[float] = None) -> dict:
    """Compute active (k1, k2, k3, vignette_amount, vignette_midpoint, remove_ca) from configuration dictionary."""
    if not isinstance(lens_cfg, dict):
        return {
            "enabled": False,
            "remove_ca": False,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "vignette_amount": 0.0,
            "vignette_midpoint": 50.0
        }

    active_tab = lens_cfg.get("active_tab", "profile")
    if active_tab == "manual":
        is_enabled = bool(lens_cfg.get("manual_enabled", True)) and (
            abs(float(lens_cfg.get("manual_distortion", 0.0))) > 0.001 or
            abs(float(lens_cfg.get("manual_vignette_amount", 0.0))) > 0.001
        )
    else:
        is_enabled = bool(lens_cfg.get("profile_enabled", lens_cfg.get("enabled", False)))

    remove_ca = bool(lens_cfg.get("remove_ca", False))

    if not is_enabled and not remove_ca:
        return {
            "enabled": False,
            "remove_ca": False,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "vignette_amount": 0.0,
            "vignette_midpoint": 50.0
        }

    k1, k2, k3 = 0.0, 0.0, 0.0
    v_amt, v_mid = 0.0, 50.0

    if is_enabled:
        if active_tab == "manual":
            m_dist = float(lens_cfg.get("manual_distortion", 0.0))
            k1, k2, k3 = compute_manual_distortion_coeffs(m_dist)
            m_v_amt = float(lens_cfg.get("manual_vignette_amount", 0.0))
            m_v_mid = float(lens_cfg.get("manual_vignette_midpoint", 0.0))
            v_amt, v_mid = compute_manual_vignette_coeffs(m_v_amt, m_v_mid)
        else:
            # Profile mode
            p_path = lens_cfg.get("profile_path", "")
            p_dist_val = float(lens_cfg.get("profile_distortion", 0.0))
            p_vig_val = float(lens_cfg.get("profile_vignette", 0.0))

            # Scale distortion: 0 -> 1.0 (100%), +100 -> 2.0 (200%), -100 -> 0.0 (0%)
            dist_scale = max(0.0, 1.0 + (p_dist_val / 100.0))

            if p_path and os.path.isfile(p_path):
                db = LCPDatabase.get_instance()
                prof = db.load_profile(p_path)
                if prof:
                    fl_parsed = _parse_float_safe(focal_length)
                    ap_parsed = _parse_float_safe(aperture)
                    calib = prof.get_best_calibration(fl_parsed, ap_parsed)
                    k1 = calib.k1 * dist_scale
                    k2 = calib.k2 * dist_scale
                    k3 = calib.k3 * dist_scale
                    if calib.vignette_coeffs:
                        v1 = calib.vignette_coeffs[0]
                        v2 = calib.vignette_coeffs[1] if len(calib.vignette_coeffs) > 1 else 0.0
                        v3 = calib.vignette_coeffs[2] if len(calib.vignette_coeffs) > 2 else 0.0
                        falloff = 1.0 + v1 + v2 + v3
                        base_amt = (1.0 / max(0.1, falloff)) - 1.0
                        base_amt = max(0.2, min(2.0, base_amt))
                    else:
                        base_amt = 0.6

                    if p_vig_val >= 0:
                        v_amt = base_amt * (1.0 + p_vig_val / 100.0)
                    else:
                        # Negative slider value:抵消去暗角并加深四角暗角
                        v_amt = base_amt + (p_vig_val / 100.0) * (base_amt + 0.8)

                    v_mid = 50.0

    return {
        "enabled": is_enabled,
        "remove_ca": remove_ca,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "vignette_amount": v_amt,
        "vignette_midpoint": v_mid
    }
