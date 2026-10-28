"""lens_correction.py - Lens Optical & Geometric Correction Engine for SpektraDarkroom
Supports:
1. Camera Raw .lcp XML profile parsing (attributes & child elements)
2. RAW / DNG / EXIF metadata extraction for automatic lens identification
3. Unified radial polynomial calculation for GPU GLSL Fragment Shader and CPU fallback
4. System-wide LensProfiles indexer with brand/model hierarchy
"""

import os
import re
import sys
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
    vignette_fx: float = 1.0
    vignette_fy: float = 1.0
    vignette_cx: float = 0.5
    vignette_cy: float = 0.5
    vignette_piecewise: List[Tuple[float, float]] = field(default_factory=list)


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
            min_fl_entry = min(self.calibrations, key=lambda c: c.focal_length)
            fl = min_fl_entry.focal_length

        ap_parsed = _parse_float_safe(aperture)
        ap = ap_parsed if ap_parsed is not None else 2.8

        best_entry = self.calibrations[0]
        best_dist = float("inf")

        for entry in self.calibrations:
            fl_diff = abs(entry.focal_length - fl)
            try:
                ap_diff = abs(math.log2(max(0.5, entry.aperture) / max(0.5, ap)))
            except Exception:
                ap_diff = abs(entry.aperture - ap)
            dist = fl_diff * 10.0 + ap_diff
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        return best_entry

    def get_best_distortion_calibration(self, focal_length: Optional[float] = None, aperture: Optional[float] = None) -> LensCalibrationEntry:
        """Find the calibration entry nearest to given focal length with calibrated distortion data."""
        dist_entries = [c for c in self.calibrations if abs(c.k1) > 1e-6 or abs(c.k2) > 1e-6 or abs(c.k3) > 1e-6]
        if not dist_entries:
            return self.get_best_calibration(focal_length, aperture)

        fl = _parse_float_safe(focal_length)
        if fl is None:
            fl = min(dist_entries, key=lambda c: c.focal_length).focal_length

        ap_parsed = _parse_float_safe(aperture)
        ap = ap_parsed if ap_parsed is not None else 2.8

        best_entry = dist_entries[0]
        best_dist = float("inf")
        for entry in dist_entries:
            fl_diff = abs(entry.focal_length - fl)
            try:
                ap_diff = abs(math.log2(max(0.5, entry.aperture) / max(0.5, ap)))
            except Exception:
                ap_diff = abs(entry.aperture - ap)
            dist = fl_diff * 10.0 + ap_diff
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        return best_entry

    def get_best_vignette_calibration(self, focal_length: Optional[float] = None, aperture: Optional[float] = None) -> Optional[LensCalibrationEntry]:
        """Find the calibration entry nearest to given focal length and aperture that contains vignette calibration data."""
        vig_entries = [c for c in self.calibrations if c.vignette_coeffs or c.vignette_piecewise]
        if not vig_entries:
            return None

        fl = _parse_float_safe(focal_length)
        if fl is None:
            fl = min(vig_entries, key=lambda c: c.focal_length).focal_length

        ap_parsed = _parse_float_safe(aperture)
        ap = ap_parsed if ap_parsed is not None else 2.8

        best_entry = vig_entries[0]
        best_dist = float("inf")
        for entry in vig_entries:
            fl_diff = abs(entry.focal_length - fl)
            try:
                ap_diff = abs(math.log2(max(0.5, entry.aperture) / max(0.5, ap)))
            except Exception:
                ap_diff = abs(entry.aperture - ap)
            dist = fl_diff * 10.0 + ap_diff * 2.0
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

            # 2. Calibration entries pass: each calibration profile is an <rdf:li> inside CameraProfiles
            calib_nodes = [elem for elem in root.iter() if elem.tag.split("}")[-1] == "li"]
            if not calib_nodes:
                calib_nodes = [
                    elem for elem in root.iter()
                    if elem.tag.split("}")[-1] == "Description" and ("FocalLength" in {k.split("}")[-1]: v for k, v in elem.attrib.items()})
                ]

            for node in calib_nodes:
                data = {}
                piecewise = []
                for sub in node.iter():
                    sub_tag = sub.tag.split("}")[-1]
                    for k, v in sub.attrib.items():
                        data[k.split("}")[-1]] = v
                    if sub.text and sub.text.strip():
                        data[sub_tag] = sub.text.strip()
                    if sub_tag == "VignetteModelPiecewiseParam":
                        for pli in sub.iter():
                            if pli.tag.split("}")[-1] == "li" and pli.text:
                                try:
                                    parts = [float(x.strip()) for x in pli.text.split(",") if x.strip()]
                                except (TypeError, ValueError):
                                    # One malformed pair must not discard the whole profile.
                                    continue
                                if len(parts) >= 2:
                                    piecewise.append((parts[0], parts[1]))

                if "FocalLength" not in data and "RadialDistortParam1" not in data and "VignetteModelParam1" not in data and not piecewise:
                    continue

                fl = _parse_float_safe(data.get("FocalLength"))
                ap = None
                if "ApertureValue" in data:
                    try:
                        raw_ap = float(data["ApertureValue"])
                        # Adobe ApertureValue is APEX: F# = 2 ^ (Av / 2)
                        ap = (2.0 ** (raw_ap / 2.0)) if raw_ap < 15.0 else raw_ap
                    except Exception:
                        pass

                fd = 3.0
                if "FocusDistance" in data:
                    try:
                        fd = float(data["FocusDistance"])
                    except Exception:
                        pass

                k1, k2, k3 = 0.0, 0.0, 0.0
                if "RadialDistortParam1" in data:
                    try:
                        k1 = float(data["RadialDistortParam1"])
                        k2 = float(data.get("RadialDistortParam2", 0.0))
                        k3 = float(data.get("RadialDistortParam3", 0.0))
                    except Exception:
                        pass
                elif "RadialDistortCoeff" in data:
                    try:
                        parts = [float(p) for p in re.split(r"[\s,]+", data["RadialDistortCoeff"].strip()) if p]
                        if len(parts) >= 1: k1 = parts[0]
                        if len(parts) >= 2: k2 = parts[1]
                        if len(parts) >= 3: k3 = parts[2]
                    except Exception:
                        pass

                vig_coeffs = []
                if "VignetteModelParam1" in data:
                    try:
                        vig_coeffs = [
                            float(data["VignetteModelParam1"]),
                            float(data.get("VignetteModelParam2", 0.0)),
                            float(data.get("VignetteModelParam3", 0.0))
                        ]
                    except Exception:
                        pass
                elif "VignetteModelCoeff" in data:
                    try:
                        vig_coeffs = [float(p) for p in re.split(r"[\s,]+", data["VignetteModelCoeff"].strip()) if p]
                    except Exception:
                        pass

                # Empty calibration fields are common in hand-made / older LCPs. Parsing
                # them with bare float() raised and threw away every calibration in the
                # file, which showed up as "this lens simply does not exist".
                fx = _parse_float_safe(data.get("FocalLengthX")) or 1.0
                fy = _parse_float_safe(data.get("FocalLengthY")) or fx
                cx = _parse_float_safe(data.get("ImageXCenter")) or 0.5
                cy = _parse_float_safe(data.get("ImageYCenter")) or 0.5

                if abs(k1) > 1e-6 or abs(k2) > 1e-6 or abs(k3) > 1e-6 or vig_coeffs or piecewise:
                    calibrations.append(LensCalibrationEntry(
                        focal_length=fl if fl is not None else 50.0,
                        aperture=ap if ap is not None else 2.8,
                        focus_distance=fd,
                        k1=k1,
                        k2=k2,
                        k3=k3,
                        vignette_coeffs=vig_coeffs,
                        vignette_fx=fx,
                        vignette_fy=fy,
                        vignette_cx=cx,
                        vignette_cy=cy,
                        vignette_piecewise=piecewise
                    ))

            if not calibrations and calib_nodes:
                logger.warning(
                    f"LCP {os.path.basename(lcp_path)} declares {len(calib_nodes)} calibration "
                    f"node(s) but none could be read; the profile will not correct anything.")
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


def get_user_lcp_directory() -> str:
    """Return the user lens profile directory (%APPDATA%/SpektraDarkroom/LensProfiles)."""
    try:
        from .path_utils import get_user_lens_profiles_dir
        return get_user_lens_profiles_dir()
    except Exception:
        if sys.platform == "win32":
            base = os.getenv("APPDATA") or os.path.expanduser("~")
            p = os.path.join(base, "SpektraDarkroom", "LensProfiles")
        else:
            xdg_data = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
            p = os.path.join(xdg_data, "spektra_darkroom", "LensProfiles")
        os.makedirs(p, exist_ok=True)
        return p


def get_bundled_lcp_directory() -> str:
    """Return bundled resources lens profile directory if it exists."""
    try:
        from .path_utils import get_resource_dir
        return os.path.join(get_resource_dir(), "lens_profiles")
    except Exception:
        return ""


def find_all_lcp_root_directories() -> List[str]:
    """Scan and aggregate user, bundled, and system Adobe directories."""
    roots = []
    # 1. User custom directory (top priority)
    user_dir = get_user_lcp_directory()
    if os.path.isdir(user_dir):
        roots.append(user_dir)

    # 2. Bundled application resources (if present)
    bundled_dir = get_bundled_lcp_directory()
    if bundled_dir and os.path.isdir(bundled_dir) and bundled_dir not in roots:
        roots.append(bundled_dir)

    # 3. System Adobe Camera Raw directories (read-only reference)
    for p in find_system_lcp_directories():
        if p not in roots:
            roots.append(p)
    return roots


def extract_lens_model_code(lens_name: str) -> str:
    """Extract lens model / series identification code.
    Examples:
        'E 28-200mm F2.8-5.6 A071' -> 'A071' (Tamron)
        'E 11-20mm F2.8 B060' -> 'B060' (Tamron)
        '35mm F1.4 DG DN | Art 021' -> 'A021' (Sigma)
        '10-18mm F2.8 DC DN | Contemporary 023' -> 'C023' (Sigma)
        'E 24mm F2.8 F051' -> 'F051' (Tamron)
    """
    if not lens_name:
        return ""
    s = str(lens_name).strip()

    # 1. Sigma Global Vision series: Art 021 -> A021, Contemporary 023 -> C023, Sport 020 -> S020
    m_art = re.search(r'\b(Art|Contemporary|Sports?)\s+(\d{3})\b', s, re.IGNORECASE)
    if m_art:
        return f"{m_art.group(1).upper()[0]}{m_art.group(2)}"

    # 2. Sigma code: C023, A018, S020
    m_sig = re.search(r'\b([CAS]\d{3}[A-Za-z]?)\b', s)
    if m_sig:
        return m_sig.group(1).upper()

    # 3. Tamron model code: A071, B060, F051, A036, etc.
    aperture_numbers = {"11", "16", "22", "32", "45", "64"}
    for m in re.finditer(r'\b([ABCFG]\d{2,3}[A-Za-z]?)\b', s):
        code = m.group(1).upper()
        # Exclude apertures like F11, F16, F22
        if code.startswith("F") and code[1:] in aperture_numbers:
            continue
        return code

    return ""


def detect_lens_brand(lens_model: str, camera_make: Optional[str] = None) -> Optional[str]:
    """Detect lens brand from lens model string, model code (e.g. A071), or fallback to camera make.
    Accurately maps abbreviations, adapters, technology tags, and model codes to database makes.
    """
    if not lens_model:
        return (camera_make or "").strip().capitalize() if camera_make else None

    lm_lower = lens_model.lower().strip()

    # 1. If an adapter is in the string (e.g. "YONGNUO ADPTER II + TAMRON 35-150mm"),
    # prioritize the primary mounted lens brand following/inside the adapter string
    if "adapter" in lm_lower or "adpter" in lm_lower or "+" in lm_lower:
        for b_name in ["tamron", "sigma", "samyang", "rokinon", "viltrox", "tokina", "zeiss", "canon", "nikon", "sony"]:
            if b_name in lm_lower:
                return b_name.capitalize()

    # 2. Explicit lens brand names (highest certainty)
    if "tamron" in lm_lower:
        return "Tamron"
    if "sigma" in lm_lower:
        return "Sigma"
    if "samyang" in lm_lower or "rokinon" in lm_lower:
        return "Samyang"
    if "viltrox" in lm_lower:
        return "Viltrox"
    if "laowa" in lm_lower or "venus" in lm_lower:
        return "Venus Optics"
    if "ttartisan" in lm_lower or "tt artisan" in lm_lower:
        return "TTArtisan"
    if "7artisans" in lm_lower or "7 artisans" in lm_lower:
        return "7Artisans"
    if "meike" in lm_lower:
        return "Meike"
    if "tokina" in lm_lower:
        return "Tokina"
    if "zeiss" in lm_lower:
        return "Zeiss"
    if "voigtlander" in lm_lower or "voigtländer" in lm_lower:
        return "Voigtlander"
    if "irix" in lm_lower:
        return "IRIX"
    if "schneider" in lm_lower:
        return "Schneider"
    if "leica" in lm_lower:
        return "Leica"
    if "canon" in lm_lower:
        return "Canon"
    if "nikon" in lm_lower or "nikkor" in lm_lower:
        return "Nikon"
    if "sony" in lm_lower:
        return "Sony"
    if "fujifilm" in lm_lower or "fuji" in lm_lower or "fujinon" in lm_lower:
        return "Fujifilm"
    if "panasonic" in lm_lower or "lumix" in lm_lower:
        return "Panasonic"
    if "olympus" in lm_lower or "m.zuiko" in lm_lower or "zuiko" in lm_lower:
        return "Olympus"
    if "pentax" in lm_lower:
        return "Pentax"
    if "ricoh" in lm_lower:
        return "Ricoh"
    if "hasselblad" in lm_lower:
        return "Hasselblad"
    if re.search(r"\b(yn|yongnuo|yongnou)\b", lm_lower) or lm_lower.startswith("yn") or "ynlens" in lm_lower:
        return "Yongnuo"

    # 3. Optical technology terms & sub-brand lines (where brand name was omitted by camera body)
    if any(k in lm_lower for k in ["rxd", "vxd", "osd", "pzd", "hld", "di iii", "di-iii", "di ii", "di-ii", "di vc", "vc usd"]) or re.search(r"\bsp\b", lm_lower):
        return "Tamron"
    if any(k in lm_lower for k in ["dg dn", "dc dn"]) or re.search(r"\b(contemporary|art|sports?)\b", lm_lower):
        return "Sigma"
    if re.search(r"\b(batis|loxia|touit|milvus|otus)\b", lm_lower):
        return "Zeiss"
    if re.search(r"\b(atx-m|atx-i|at-x|firin|opera)\b", lm_lower):
        return "Tokina"

    # 4. Lens model code detection (e.g. 'A071', 'B060', 'C023', 'A021')
    code = extract_lens_model_code(lens_model)
    if code:
        if re.match(r"^[ABCFG]\d{2,3}[A-Z]?$", code):
            if re.match(r"^[CS]\d{3}", code):
                return "Sigma"
            return "Tamron"

    # 5. OEM camera manufacturer series terms
    if re.search(r"\b(fe|sel\d+|gm|g master)\b", lm_lower):
        return "Sony"
    if re.search(r"\b(ef-s|ef-m|\bef\b|\brf\b|\brf-s\b)\b", lm_lower):
        return "Canon"
    if re.search(r"\b(af-s|af-p|z-mount|\bvr\b)\b", lm_lower):
        return "Nikon"
    if re.search(r"\b(xf|xc|gf)\b", lm_lower):
        return "Fujifilm"

    if camera_make and camera_make.strip():
        return camera_make.strip().capitalize()
    return None


def extract_lcp_header(lcp_path: str) -> Tuple[str, str]:
    """Quickly extract (make, lens_name) from LCP XML header without full parsing."""
    make = ""
    lens_name = ""
    if not os.path.isfile(lcp_path):
        return make, lens_name
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(lcp_path)
        root = tree.getroot()
        for elem in root.iter():
            attribs = {k.split("}")[-1]: v for k, v in elem.attrib.items()}
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if not lens_name:
                lens_name = attribs.get("LensPrettyName") or attribs.get("Lens") or attribs.get("Model") or (elem.text.strip() if tag in ("LensPrettyName", "Lens", "Model") and elem.text else "")
            if not make:
                make = attribs.get("Make") or attribs.get("CameraPrettyName") or (elem.text.strip() if tag in ("Make", "CameraPrettyName") and elem.text else "")
            if make and lens_name:
                break

        brand_cand = detect_lens_brand(lens_name) or detect_lens_brand(os.path.basename(lcp_path))
        if brand_cand:
            make = brand_cand
    except Exception:
        pass
    return make.strip(), lens_name.strip()


def import_user_lcp_file(source_path: str, target_make: Optional[str] = None) -> Tuple[bool, str, str, str, str]:
    """Import a .lcp file into user lens profile directory (%APPDATA%/SpektraDarkroom/LensProfiles/<Make>/).

    Returns:
        (success, make, model_name, profile_display_name, dest_file_path)
    """
    if not source_path or not os.path.isfile(source_path):
        return False, "", "", "", ""

    import shutil
    parsed_make, parsed_lens = extract_lcp_header(source_path)

    make = ""
    if target_make and target_make.strip() and target_make.strip() != "无":
        make = target_make.strip()
    elif parsed_make:
        make = parsed_make
    else:
        make = "Custom"

    user_dir = get_user_lcp_directory()
    dest_dir = os.path.join(user_dir, make)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
    except Exception as e:
        logger.error(f"Failed to copy LCP file to {dest_dir}: {e}")
        return False, "", "", "", ""

    db = LCPDatabase.get_instance()
    db.index_installed_profiles()

    f = os.path.basename(dest_path)
    clean_name = f[:-4]
    if clean_name.upper().startswith(make.upper()):
        clean_name = clean_name[len(make):].strip(" -_()")
    if clean_name.endswith(" - RAW"):
        clean_name = clean_name[:-6].strip()
    elif clean_name.endswith(" - JPEG"):
        clean_name = clean_name[:-7].strip()
    model_name = clean_name or f[:-4]
    prof_display = parsed_lens or f[:-4]

    return True, make, model_name, prof_display, dest_path


def copy_system_acr_profiles_to_user_dir(progress_cb=None) -> Tuple[int, int]:
    """Copy all .lcp files from system Adobe Camera Raw directories into the user AppData directory.
    Returns (num_copied, num_skipped).
    """
    import shutil
    acr_dirs = find_system_lcp_directories()
    user_dir = get_user_lcp_directory()
    if not acr_dirs:
        return 0, 0

    copied = 0
    skipped = 0
    for root_dir in acr_dirs:
        for make in os.listdir(root_dir):
            make_src = os.path.join(root_dir, make)
            if not os.path.isdir(make_src):
                continue
            make_dest = os.path.join(user_dir, make)
            os.makedirs(make_dest, exist_ok=True)
            for r, dirs, files in os.walk(make_src):
                for f in files:
                    if f.lower().endswith(".lcp"):
                        src_f = os.path.join(r, f)
                        dest_f = os.path.join(make_dest, f)
                        if os.path.exists(dest_f):
                            skipped += 1
                            continue
                        try:
                            shutil.copy2(src_f, dest_f)
                            copied += 1
                            if progress_cb and copied % 50 == 0:
                                progress_cb(copied)
                        except Exception:
                            pass

    db = LCPDatabase.get_instance()
    db.index_installed_profiles()
    return copied, skipped


class LCPDatabase:
    """In-memory indexer and manager for Lens Profiles (.lcp)."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "LCPDatabase":
        if cls._instance is None:
            cls._instance = LCPDatabase()
        return cls._instance

    def __init__(self):
        self._roots: List[str] = find_all_lcp_root_directories()
        self._makes: List[str] = []
        self._models_by_make: Dict[str, List[Tuple[str, str, str]]] = {}  # make -> [(model_name, profile_name, path)]
        self._profile_cache: Dict[str, LensProfile] = {}
        self._indexed = False
        self._build_index()

    def index_installed_profiles(self):
        """Build or refresh profile database index."""
        self._roots = find_all_lcp_root_directories()
        self._models_by_make.clear()
        # A re-imported .lcp with the same name must not keep serving the previous profile.
        self._profile_cache.clear()
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
        seen_paths = set()
        for root_dir in self._roots:
            make_dir = os.path.join(root_dir, make)
            if not os.path.isdir(make_dir):
                continue
            for r, dirs, files in os.walk(make_dir):
                for f in files:
                    if f.lower().endswith(".lcp"):
                        p = os.path.join(r, f)
                        if p in seen_paths:
                            continue
                        seen_paths.add(p)
                        # Derive model name cleanly from filename or quick metadata
                        clean_name = f[:-4]
                        # Remove leading brand prefix if redundant
                        if clean_name.upper().startswith(make.upper()):
                            clean_name = clean_name[len(make):].strip(" -_()")
                        if clean_name.endswith(" - RAW"):
                            clean_name = clean_name[:-6].strip()
                        elif clean_name.endswith(" - JPEG"):
                            clean_name = clean_name[:-7].strip()
                        if not clean_name:
                            clean_name = f[:-4]
                        prof_display = f[:-4]
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
        """Attempt to find matching (make, model, profile_name, lcp_path) from EXIF metadata.
        Uses intelligent brand deduction, model code extraction (e.g. Tamron A071, Sigma Art 021),
        mount alignment, and token overlap scoring to reliably identify the correct Adobe LCP profile.
        """
        if not self._makes:
            return None

        clean_lens = (lens_model or "").strip()
        clean_cam_make = (camera_make or "").strip()
        detected_brand = detect_lens_brand(clean_lens, clean_cam_make)
        code = extract_lens_model_code(clean_lens).lower()

        # 1. Determine make candidates
        make_candidates = []
        if detected_brand:
            for m in self._makes:
                if m.lower() == detected_brand.lower():
                    make_candidates.append(m)
                    break

        # Check if lens_model starts with or contains any known make
        for m in self._makes:
            if m.lower() in clean_lens.lower() and m not in make_candidates:
                make_candidates.append(m)

        # Fallback candidate to camera manufacturer
        if clean_cam_make:
            for m in self._makes:
                if m.lower() == clean_cam_make.lower() and m not in make_candidates:
                    make_candidates.append(m)

        if not make_candidates and self._makes:
            make_candidates = list(self._makes)

        # 2. Search inside candidate makes with advanced alphanumeric tokenization
        lens_tokens = set(re.findall(r"[a-z0-9]+", clean_lens.lower()))
        noisy_tokens = {"mm", "f", "lens", "adapter", "adpter", "mount"}
        lens_tokens = {t for t in lens_tokens if t not in noisy_tokens}
        best_match = None
        best_score = -1

        for m in make_candidates:
            models = self.get_models_for_make(m)
            is_detected_brand = bool(detected_brand and m.lower() == detected_brand.lower())

            for mod_name, prof_name, p in models:
                p_leaf = os.path.basename(p).lower()
                mod_tokens = set(re.findall(r"[a-z0-9]+", mod_name.lower()))
                mod_tokens = {t for t in mod_tokens if t not in noisy_tokens}
                common = lens_tokens.intersection(mod_tokens)
                score = len(common) * 2

                # Brand alignment bonus
                if is_detected_brand:
                    score += 5

                # Mount / Camera make alignment bonus
                if clean_cam_make and clean_cam_make.lower() in (mod_name.lower() + " " + p_leaf):
                    score += 3

                # Model code match (HIGHEST PRIORITY!)
                if code and (code in mod_name.lower() or code in p_leaf):
                    score += 15

                # Prefer RAW profile if available
                if p_leaf.endswith(" - raw.lcp"):
                    score += 1

                if score > best_score and score >= 4:
                    best_score = score
                    best_match = (m, mod_name, prof_name, p)

        return best_match


def read_lens_metadata(file_path: str) -> Dict[str, any]:
    """Extract lens, camera, focal length, aperture, and metadata from RAW or standard image file."""
    meta = {
        "camera_make": "",
        "camera_model": "",
        "lens_make": "",
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
                    meta["focal_length"] = getattr(other_obj, "focal_length", None) or getattr(other_obj, "focal_len", None)
                    meta["aperture"] = getattr(other_obj, "aperture", None)
                    meta["iso"] = getattr(other_obj, "iso_speed", None)
        except Exception as e:
            logger.debug(f"rawpy metadata read error for {file_path}: {e}")

    # 1.5. Try tifffile if camera make/model or lens is still empty (e.g. ARW, DNG, CR2, TIFF)
    if not meta["camera_make"] or not meta["camera_model"] or not meta["lens_model"]:
        try:
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                for page in tif.pages:
                    for tag in page.tags:
                        t_name = tag.name
                        if t_name == "Make" and not meta["camera_make"]:
                            meta["camera_make"] = str(tag.value).strip()
                        elif t_name == "Model" and not meta["camera_model"]:
                            meta["camera_model"] = str(tag.value).strip()
                        elif (t_name in ("LensModel", 0xa434)) and not meta["lens_model"]:
                            meta["lens_model"] = str(tag.value).strip().strip("\x00")
        except Exception as e:
            logger.debug(f"tifffile metadata read error for {file_path}: {e}")

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

    if not meta.get("lens_make") and meta.get("lens_model"):
        meta["lens_make"] = detect_lens_brand(meta["lens_model"], meta.get("camera_make")) or ""

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


def evaluate_vignette_gain(calib: Optional[LensCalibrationEntry], photo_aperture: Optional[float] = None, aspect_ratio: float = 0.6667) -> float:
    """Evaluate natural lens vignette compensation factor (base_amt) from calibration data.

    Returns:
        float: base_amt (typically 0.10 ~ 0.35, max 0.45) representing fractional gain
               applied at the outermost image corners to naturally cancel light falloff.
    """
    if not calib:
        return 0.15

    raw_gain = 0.15

    if calib.vignette_piecewise:
        # Last point in piecewise curve represents maximum radial distance / corner
        last_pt = calib.vignette_piecewise[-1]
        v_corner = max(0.25, min(1.0, float(last_pt[1])))
        raw_gain = (1.0 / v_corner) - 1.0
    elif calib.vignette_coeffs:
        v1 = calib.vignette_coeffs[0]
        v2 = calib.vignette_coeffs[1] if len(calib.vignette_coeffs) > 1 else 0.0
        v3 = calib.vignette_coeffs[2] if len(calib.vignette_coeffs) > 2 else 0.0

        fx = max(0.1, getattr(calib, "vignette_fx", 1.0))
        fy = max(0.1, getattr(calib, "vignette_fy", fx))

        # Normalized coordinates at image corner (3:2 standard sensor: dx = 0.5, dy = 0.5 * 2/3 = 0.3333)
        dx = 0.5 / fx
        dy = (0.5 * aspect_ratio) / fy
        r2 = dx * dx + dy * dy
        r4 = r2 * r2
        r6 = r4 * r2

        v_corner = 1.0 + v1 * r2 + v2 * r4 + v3 * r6
        # Clamp to realistic optical transmittance [0.25, 1.0]
        v_clamped = max(0.25, min(1.0, v_corner))
        raw_gain = (1.0 / v_clamped) - 1.0

    # Scale down if photo is stopped down beyond calibration entry
    ap_photo = _parse_float_safe(photo_aperture)
    if ap_photo and ap_photo > calib.aperture and calib.aperture > 0.5:
        try:
            stops = math.log2(ap_photo / calib.aperture)
            # Physical vignetting reduces by ~25% per stop
            scale = max(0.35, 1.0 - stops * 0.25)
            raw_gain *= scale
        except Exception:
            pass

    # Natural optical compensation: 65% of raw physical drop (prevents reverse vignetting / white halo)
    eff_gain = raw_gain * 0.65
    return max(0.05, min(0.45, eff_gain))


def compute_effective_lens_params(lens_cfg: dict, focal_length: Optional[float] = None, aperture: Optional[float] = None) -> dict:
    """Compute active (k1, k2, k3, vignette_amount, vignette_midpoint, remove_ca) from configuration dictionary."""
    if not isinstance(lens_cfg, dict):
        return {
            "enabled": False,
            "remove_ca": False,
            "ca_kr": 0.0,
            "ca_kb": 0.0,
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
            "ca_kr": 0.0,
            "ca_kb": 0.0,
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

                    calib_dist = prof.get_best_distortion_calibration(fl_parsed, ap_parsed)

                    # Adobe ACM polynomial defines r = R_sensor / eff_fl.
                    # Our shader operates on corner-normalized coordinates r_norm in [0, 1] where r_norm = 1 at the image corner.
                    # Standard 35mm full frame sensor corner semi-diagonal: sqrt(18^2 + 12^2) = 21.633333 mm.
                    # For a given crop factor (sensor format factor) C: R_corner = 21.633333 / C.
                    # At the corner: r_corner = (21.633333 / C) / eff_fl.
                    # Thus r = r_norm * r_corner.
                    # Since scale = 1 + k1*r^2 + k2*r^4 + k3*r^6:
                    # k1*r^2 = (k1 * r_corner^2) * r_norm^2
                    # k2*r^4 = (k2 * r_corner^4) * r_norm^4
                    # k3*r^6 = (k3 * r_corner^6) * r_norm^6
                    eff_fl = fl_parsed if (fl_parsed and fl_parsed > 0) else calib_dist.focal_length
                    if not eff_fl or eff_fl <= 0:
                        eff_fl = 50.0
                    crop_factor = float(prof.sensor_format_factor) if (prof.sensor_format_factor and prof.sensor_format_factor > 0) else 1.0
                    r_corner = 21.633333 / (crop_factor * eff_fl)

                    k1 = calib_dist.k1 * (r_corner ** 2) * dist_scale
                    k2 = calib_dist.k2 * (r_corner ** 4) * dist_scale
                    k3 = calib_dist.k3 * (r_corner ** 6) * dist_scale

                    calib_vig = prof.get_best_vignette_calibration(fl_parsed, ap_parsed)
                    base_amt = evaluate_vignette_gain(calib_vig, photo_aperture=ap_parsed)

                    if p_vig_val >= 0:
                        v_amt = base_amt * (1.0 + p_vig_val / 100.0)
                    else:
                        # Negative slider value:抵消去暗角并加深四角暗角
                        v_amt = base_amt + (p_vig_val / 100.0) * (base_amt + 0.5)

                    v_mid = 50.0

    return {
        "enabled": is_enabled,
        "remove_ca": remove_ca,
        "ca_kr": float(lens_cfg.get("ca_kr", 0.0)) if isinstance(lens_cfg, dict) else 0.0,
        "ca_kb": float(lens_cfg.get("ca_kb", 0.0)) if isinstance(lens_cfg, dict) else 0.0,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "vignette_amount": v_amt,
        "vignette_midpoint": v_mid
    }
