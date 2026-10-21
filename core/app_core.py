import os
import json
import base64
import time
import io
import threading
import uuid
import numpy as np
import cv2
import logging
from PIL import Image

logger = logging.getLogger("SpektraEngine")

import spektrafilm
from spektrafilm.runtime.api import init_params, digest_params
from spektrafilm.runtime.process import Simulator
from spektrafilm.profiles import load_processed_profile

try:
    import rawpy
except ImportError:
    rawpy = None

from .format_constants import (
    RAW_EXTENSIONS,
    STANDARD_IMAGE_EXTENSIONS,
    ALL_SUPPORTED_IMAGE_EXTENSIONS
)


# Color matrices for accurate gamut transitions
# ACES2065-1 (AP0) to ProPhoto RGB
ACES_TO_PROPHOTO = np.array([
    [ 1.23938034, -0.16396782, -0.07523338],
    [ 0.00361136,  1.08961365, -0.09326579],
    [-0.00205968, -0.00225159,  1.00458558]
], dtype=np.float32)

# sRGB (linear) to ProPhoto RGB
SRGB_TO_PROPHOTO = np.array([
    [0.52882410, 0.33406099, 0.13736169],
    [0.09752941, 0.87900741, 0.02339812],
    [0.01635990, 0.10661249, 0.87724852]
], dtype=np.float32)

# ProPhoto RGB to sRGB (linear)
PROPHOTO_TO_SRGB = np.array([
    [ 2.03649172, -0.73759065, -0.29925987],
    [-0.22571798,  1.22317653,  0.00272522],
    [-0.01054513, -0.13487985,  1.14521015]
], dtype=np.float32)


# Log Shaper constants: maps scene-linear HDR [0.0, 8.0] into [0.0, 1.0] with dense shadow resolution
SHAPER_C = 60.0
SHAPER_MAX = 8.0
SHAPER_S = float(np.log(1.0 + SHAPER_C * SHAPER_MAX))  # ~6.175867147775916
SHAPER_INV_S = float(1.0 / SHAPER_S)                    # ~0.16192058097945077


def shaper_forward(lin):
    """Map scene-linear HDR [0.0, 8.0+] into [0.0, 1.0] with logarithmic shadow allocation."""
    c = np.maximum(0.0, lin)
    return (np.log(1.0 + SHAPER_C * c) * SHAPER_INV_S).astype(np.float32)


def shaper_inverse(t):
    """Map shaper lattice coordinates [0.0, 1.0] into scene-linear HDR [0.0, 8.0]."""
    return ((np.exp(t * SHAPER_S) - 1.0) / SHAPER_C).astype(np.float32)


def srgb_to_linear(srgb):
    """Decode standard sRGB non-linear CCTF into linear values."""
    srgb_c = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb_c <= 0.04045, srgb_c / 12.92, np.power(np.maximum(0.0, (srgb_c + 0.055) / 1.055), 2.4)).astype(np.float32)


def srgb_to_prophoto(srgb):
    """Convert standard non-linear Display sRGB [0.0, 1.0] to Linear ProPhoto RGB."""
    lin_srgb = srgb_to_linear(srgb)
    return np.maximum(0.0, lin_srgb @ SRGB_TO_PROPHOTO.T).astype(np.float32)


def prophoto_to_srgb(lin_pro):
    """Convert Linear ProPhoto RGB to standard Display sRGB."""
    lin_srgb = np.maximum(0.0, lin_pro @ PROPHOTO_TO_SRGB.T)
    srgb_disp = np.where(
        lin_srgb <= 0.0031308,
        12.92 * lin_srgb,
        1.055 * np.power(np.maximum(lin_srgb, 1e-6), 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb_disp, 0.0, 1.0).astype(np.float32)


def prophoto_to_srgb_filmic(lin_pro):
    """Convert Linear ProPhoto RGB [0.0, 8.0] to Display sRGB with smooth highlight roll-off (Bypass mode)."""
    lin_srgb = np.maximum(0.0, lin_pro @ PROPHOTO_TO_SRGB.T)
    knee = 0.5
    scale = 3.5
    comp = np.where(lin_srgb <= knee, lin_srgb, knee + (1.0 - knee) * (lin_srgb - knee) / ((lin_srgb - knee) + (scale - 1.0)))
    comp = np.clip(comp, 0.0, 1.0)
    srgb_disp = np.where(
        comp <= 0.0031308,
        12.92 * comp,
        1.055 * np.power(np.maximum(comp, 1e-6), 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb_disp, 0.0, 1.0).astype(np.float32)


def filmic_shoulder_np(c, threshold=0.75):
    """Filmic soft highlight roll-off (smooth C1 shoulder compression above threshold)."""
    over = np.maximum(0.0, c - threshold)
    span = 1.0 - threshold
    return (np.minimum(c, threshold) + span * (over / (span + over))).astype(np.float32)


def apply_highlights_shadows_np(linear_img, highlights, shadows):
    """Pre-LUT Scene-Linear Highlights & Shadows Tone Shaping (Zone System EV Hermite Spline).
    Strictly aligned with GLSL Shader & export pipeline.
    """
    hl_val = float(highlights)
    sh_val = float(shadows)
    if linear_img is None or (abs(hl_val) <= 0.001 and abs(sh_val) <= 0.001):
        return linear_img

    out = linear_img.copy()
    y_lum = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    mask = (y_lum > 1e-5)
    z = np.zeros_like(y_lum)
    z[mask] = np.log2(y_lum[mask] / 0.18)

    w_sh = np.clip(-z * 0.25, 0.0, 1.0)
    w_sh = w_sh * w_sh * (3.0 - 2.0 * w_sh)

    w_hl = np.clip(z * 0.25, 0.0, 1.0)
    w_hl = w_hl * w_hl * (3.0 - 2.0 * w_hl)

    delta_z = (sh_val * 0.02) * w_sh + (hl_val * 0.02) * w_hl
    y_target = 0.18 * (2.0 ** (z + delta_z))

    scale = np.ones_like(y_lum)
    scale[mask] = y_target[mask] / y_lum[mask]
    out *= scale[..., None]
    return out


def apply_diffusion_np(linear_img, diff_family="none", diff_strength=0.0, diff_warmth=0.0):
    """Optical Diffusion Filter (Black Pro-Mist, Glimmerglass, Pro-Mist, CineBloom).
    Applied in scene-linear radiance space before 3D LUT.
    Strictly aligned with GLSL fragment shader Vogel spiral bloom & export pipeline.
    """
    diff_fam = str(diff_family).lower()
    strength = float(diff_strength)
    if linear_img is None or diff_fam in ("none", "") or strength <= 0.001:
        return linear_img

    out = linear_img.copy()
    eff_type = 1
    if "glimmer" in diff_fam:
        eff_type = 2
    elif "cine" in diff_fam:
        eff_type = 4
    elif "pro_mist" in diff_fam or "promist" in diff_fam:
        eff_type = 3 if "black" not in diff_fam else 1

    base_rad = 0.012 * max(0.20, strength)
    if eff_type == 1:
        thr, knee, rad_mult, intensity, halo_w = 0.36, 0.16, 1.0, 1.05, 0.65
    elif eff_type == 2:
        thr, knee, rad_mult, intensity, halo_w = 0.42, 0.14, 0.75, 0.85, 0.0
    elif eff_type == 3:
        thr, knee, rad_mult, intensity, halo_w = 0.24, 0.20, 1.45, 1.25, 0.40
    elif eff_type == 4:
        thr, knee, rad_mult, intensity, halo_w = 0.20, 0.22, 1.85, 1.40, 0.85
    else:
        thr, knee, rad_mult, intensity, halo_w = 0.36, 0.16, 1.0, 1.05, 0.65

    H, W = out.shape[:2]
    luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    lo = max(0.001, thr - knee)
    hi = thr + knee
    t = np.clip((luma - lo) / max(0.0001, hi - lo), 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)
    hl = smooth_t * np.maximum(0.0, luma - lo)
    hl_factor = (hl / np.maximum(0.0001, luma))[..., None]
    hl_img = out * hl_factor

    sigma_px = max(1.0, base_rad * rad_mult * max(H, W) * 0.18)
    ksize = int(sigma_px * 3.0) * 2 + 1
    ksize = min(ksize, 301)
    bloom = cv2.GaussianBlur(hl_img, (ksize, ksize), sigma_px)

    eff_warmth = float(np.clip(halo_w + float(diff_warmth), -1.5, 1.5))
    diff_tint = np.array([
        1.0 + eff_warmth * 0.35,
        1.0 + eff_warmth * 0.08,
        1.0 - eff_warmth * 0.35
    ], dtype=np.float32)

    out += bloom * (strength * intensity) * diff_tint
    if eff_type == 3:
        out = out * 0.80 + (out + 0.015 * strength) * 0.20
    elif eff_type == 4:
        out = out * 0.75 + (out + 0.022 * strength) * 0.25

    return out


def convert_image_colorspace(img_rgb_float, target_colorspace="sRGB"):
    """Convert an image in sRGB non-linear color space [0, 1] to target color space."""
    target = str(target_colorspace).strip()
    if target in ("sRGB", "srgb", "", None):
        return np.clip(img_rgb_float, 0.0, 1.0)

    try:
        import colour
        t_low = target.lower()
        if "p3" in t_low:
            out = colour.RGB_to_RGB(img_rgb_float, 'sRGB', 'Display P3')
        elif "adobe" in t_low:
            out = colour.RGB_to_RGB(img_rgb_float, 'sRGB', 'Adobe RGB (1998)')
        elif "prophoto" in t_low:
            out = colour.RGB_to_RGB(img_rgb_float, 'sRGB', 'ProPhoto RGB')
        else:
            out = img_rgb_float
        return np.clip(out.astype(np.float32), 0.0, 1.0)
    except Exception as e:
        print(f"[SpektraEngine] Colorspace conversion error ({target}): {e}")
        return np.clip(img_rgb_float, 0.0, 1.0)


def get_icc_profile_bytes(target_colorspace="sRGB"):
    """Retrieve standard embedded ICC profile bytes for target colorspace."""
    from .path_utils import get_resource_dir
    target = str(target_colorspace).strip().lower()
    icc_dir = os.path.join(get_resource_dir(), "icc")

    cand_file = None
    if "p3" in target:
        cand_file = os.path.join(icc_dir, "DisplayP3.icc")
    elif "adobe" in target:
        cand_file = os.path.join(icc_dir, "AdobeRGB1998.icc")
    elif "prophoto" in target:
        cand_file = os.path.join(icc_dir, "ProPhotoRGB.icc")
    else:
        cand_file = os.path.join(icc_dir, "sRGB.icc")

    if cand_file and os.path.exists(cand_file):
        try:
            with open(cand_file, "rb") as f:
                return f.read()
        except Exception:
            pass
    return None


def _force_exif_orientation_upright(exif_bytes):
    """Guarantees that embedded EXIF byte stream has Orientation tag (0x0112) set to 1 (Normal/Upright).
    Since SpektraDarkroom pixel buffers are already fully demosaiced and oriented upright,
    preserving non-1 EXIF orientation flags causes external photo viewers to rotate the image
    a second time (e.g. rotating portrait negatives horizontally).
    """
    if not exif_bytes or len(exif_bytes) < 14:
        return exif_bytes
    try:
        import struct
        b = bytearray(exif_bytes)
        start = 6 if b.startswith(b'Exif\x00\x00') else 0
        if b[start:start+2] == b'II':
            endian = '<'
        elif b[start:start+2] == b'MM':
            endian = '>'
        else:
            return exif_bytes

        magic = struct.unpack_from(endian + 'H', b, start + 2)[0]
        if magic != 42:
            return exif_bytes

        ifd0_offset = struct.unpack_from(endian + 'I', b, start + 4)[0]
        pos = start + ifd0_offset
        if pos + 2 > len(b):
            return exif_bytes

        num_tags = struct.unpack_from(endian + 'H', b, pos)[0]
        pos += 2
        for _ in range(num_tags):
            if pos + 12 > len(b):
                break
            tag = struct.unpack_from(endian + 'H', b, pos)[0]
            if tag == 0x0112:
                struct.pack_into(endian + 'H', b, pos + 8, 1)
                struct.pack_into(endian + 'H', b, pos + 10, 0)
                break
            pos += 12
        return bytes(b)
    except Exception:
        return exif_bytes


UNSIGNED_RATIONAL_TAGS = {
    # SubIFD (0x8769)
    0x829A,  # ExposureTime (33434)
    0x829D,  # FNumber (33437)
    0x9102,  # CompressedBitsPerPixel (37122)
    0x9202,  # ApertureValue (37378)
    0x9205,  # MaxApertureValue (37381)
    0x9206,  # SubjectDistance (37382)
    0x920A,  # FocalLength (37386)
    0xA404,  # DigitalZoomRatio (41988)
    # IFD0
    0x011A,  # XResolution (282)
    0x011B,  # YResolution (283)
    # GPS (0x8825)
    6,       # GPSAltitude
    11,      # GPSDOP
    13,      # GPSSpeed
    15,      # GPSTrack
    17,      # GPSImgDirection
    24,      # GPSDestBearing
    26,      # GPSDestDistance
    31,      # GPSHPositioningError
}

SIGNED_RATIONAL_TAGS = {
    0x9201,  # ShutterSpeedValue (37377)
    0x9203,  # BrightnessValue (37379)
    0x9204,  # ExposureBiasValue (37380)
}


def _normalize_exif_val(cid, val):
    """Normalize raw metadata tag values (especially from tifffile tuples) into standard
    types that PIL Image.Exif() correctly serializes into valid EXIF 2.32 specifications.
    Ensures rational tags serialize as Type 5 (RATIONAL) or Type 10 (SRATIONAL) rather than Type 3 (SHORT).
    """
    from PIL import TiffImagePlugin
    if val is None:
        return None

    # 1. LensSpecification (0xA432 / 42034) - 4 unsigned rationals (min_focal, max_focal, min_f, max_f)
    if cid in (0xA432, 42034, 'LensSpecification'):
        if isinstance(val, (tuple, list)):
            if len(val) == 8:
                return tuple(
                    TiffImagePlugin.IFDRational(int(val[i]), int(val[i + 1])) if val[i + 1] != 0 else TiffImagePlugin.IFDRational(0, 1)
                    for i in range(0, 8, 2)
                )
            elif len(val) == 4:
                res = []
                for x in val:
                    if isinstance(x, (tuple, list)) and len(x) == 2:
                        res.append(TiffImagePlugin.IFDRational(int(x[0]), int(x[1])) if x[1] != 0 else TiffImagePlugin.IFDRational(0, 1))
                    elif hasattr(x, 'numerator') and hasattr(x, 'denominator'):
                        res.append(TiffImagePlugin.IFDRational(int(x.numerator), int(x.denominator)))
                    elif isinstance(x, (int, float)) and not isinstance(x, bool):
                        from fractions import Fraction
                        frac = Fraction(float(x)).limit_denominator(1000)
                        res.append(TiffImagePlugin.IFDRational(frac.numerator, frac.denominator))
                    else:
                        res.append(x)
                return tuple(res)
        return val

    # 2. Known Signed Rational tags (Type 10: SRATIONAL)
    if cid in SIGNED_RATIONAL_TAGS or cid in ('ExposureBiasValue', 'BrightnessValue', 'ShutterSpeedValue', 'CameraElevationAngle'):
        if isinstance(val, (tuple, list)) and len(val) == 2:
            num, den = int(val[0]), int(val[1])
            return TiffImagePlugin.IFDRational(num, den if den != 0 else 1)
        elif hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            return TiffImagePlugin.IFDRational(int(val.numerator), int(val.denominator))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            from fractions import Fraction
            frac = Fraction(float(val)).limit_denominator(10000)
            return TiffImagePlugin.IFDRational(frac.numerator, frac.denominator)

    # 3. Known Unsigned Rational tags (Type 5: RATIONAL)
    if cid in UNSIGNED_RATIONAL_TAGS or cid in (
        'ExposureTime', 'FNumber', 'FocalLength', 'MaxApertureValue', 'DigitalZoomRatio',
        'SubjectDistance', 'CompressedBitsPerPixel', 'XResolution', 'YResolution'
    ):
        if isinstance(val, (tuple, list)) and len(val) == 2:
            num, den = int(val[0]), int(val[1])
            return TiffImagePlugin.IFDRational(num, den if den != 0 else 1)
        elif hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            return TiffImagePlugin.IFDRational(int(val.numerator), int(val.denominator))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            from fractions import Fraction
            frac = Fraction(float(val)).limit_denominator(10000)
            return TiffImagePlugin.IFDRational(frac.numerator, frac.denominator)

    # 4. Generic 2-element integer tuple (fallback for tifffile rational representation)
    if isinstance(val, (tuple, list)) and len(val) == 2 and all(isinstance(x, (int, np.integer)) for x in val):
        num, den = int(val[0]), int(val[1])
        return TiffImagePlugin.IFDRational(num, den if den != 0 else 1)

    # 5. String tags: strip surrounding whitespace and internal/trailing NUL bytes
    if isinstance(val, str):
        return val.strip('\x00').strip()

    # 6. ExifVersion (0x9000): ensure ASCII bytes e.g. b'0232'
    if cid in (0x9000, 36864, 'ExifVersion'):
        if isinstance(val, str):
            return val.encode('ascii')

    return val


def _normalize_gps_val(gid, gv):
    """Normalize GPS metadata tag values to standard PIL IFDRational tuples."""
    from PIL import TiffImagePlugin
    if gv is None:
        return None

    # Lat, Lon, TimeStamp, DestLat, DestLon (3 unsigned rationals)
    if gid in (2, 4, 7, 20, 22) or gid in ('GPSLatitude', 'GPSLongitude', 'GPSTimeStamp', 'GPSDestLatitude', 'GPSDestLongitude'):
        if isinstance(gv, (tuple, list)):
            if len(gv) == 6:
                return (
                    TiffImagePlugin.IFDRational(int(gv[0]), int(gv[1])) if gv[1] != 0 else TiffImagePlugin.IFDRational(0, 1),
                    TiffImagePlugin.IFDRational(int(gv[2]), int(gv[3])) if gv[3] != 0 else TiffImagePlugin.IFDRational(0, 1),
                    TiffImagePlugin.IFDRational(int(gv[4]), int(gv[5])) if gv[5] != 0 else TiffImagePlugin.IFDRational(0, 1),
                )
            elif len(gv) == 3:
                res = []
                for x in gv:
                    if isinstance(x, (tuple, list)) and len(x) == 2:
                        res.append(TiffImagePlugin.IFDRational(int(x[0]), int(x[1])) if x[1] != 0 else TiffImagePlugin.IFDRational(0, 1))
                    elif hasattr(x, 'numerator') and hasattr(x, 'denominator'):
                        res.append(TiffImagePlugin.IFDRational(int(x.numerator), int(x.denominator)))
                    elif isinstance(x, (int, float)) and not isinstance(x, bool):
                        from fractions import Fraction
                        frac = Fraction(float(x)).limit_denominator(10000)
                        res.append(TiffImagePlugin.IFDRational(frac.numerator, frac.denominator))
                    else:
                        res.append(x)
                return tuple(res)
    # Altitude, Speed, Track, ImgDirection, DestBearing, DestDistance, DOP, PositioningError (1 unsigned rational)
    elif gid in (6, 11, 13, 15, 17, 24, 26, 31) or gid in ('GPSAltitude', 'GPSDOP', 'GPSSpeed', 'GPSTrack', 'GPSImgDirection'):
        if isinstance(gv, (tuple, list)) and len(gv) == 2:
            return TiffImagePlugin.IFDRational(int(gv[0]), int(gv[1])) if gv[1] != 0 else TiffImagePlugin.IFDRational(0, 1)
        elif hasattr(gv, 'numerator') and hasattr(gv, 'denominator'):
            return TiffImagePlugin.IFDRational(int(gv.numerator), int(gv.denominator))
        elif isinstance(gv, (int, float)) and not isinstance(gv, bool):
            from fractions import Fraction
            frac = Fraction(float(gv)).limit_denominator(10000)
            return TiffImagePlugin.IFDRational(frac.numerator, frac.denominator)
    elif gid == 5 and isinstance(gv, (int, np.integer)):
        return int(gv)
    elif isinstance(gv, str):
        return gv.strip('\x00').strip()
    return gv


def extract_exif_bundle(source_path):
    """Extract complete camera metadata from original photo or RAW negative (ARW, CR2, NEF, DNG, RAF, JPG, TIF, PNG).
    Returns (pil_exif, exif_bytes, tiff_tags) for embedding into exported JPEG, TIFF, and PNG images.
    """
    if not source_path or not os.path.exists(source_path):
        return None, None, []

    from PIL import Image, ExifTags, TiffImagePlugin
    import io, struct
    ext = os.path.splitext(source_path)[1].lower()

    tag_name_to_id = {v: k for k, v in ExifTags.TAGS.items()}
    gps_name_to_id = {v: k for k, v in ExifTags.GPSTAGS.items()}

    exif_bytes = None
    loaded_exif = None

    # 1. Native TIFF/RAW deep tag extraction
    # Extracts complete IFD0, ExifIFD (0x8769), and GPSTag (0x8825) directly from camera negative
    if ext in RAW_EXTENSIONS or ext in ('.tif', '.tiff'):
        try:
            import tifffile
            with tifffile.TiffFile(source_path) as tf:
                loaded_exif = Image.Exif()
                sub_exif = loaded_exif.get_ifd(0x8769)
                gps_exif = loaded_exif.get_ifd(0x8825)

                for p in tf.pages:
                    for tag in p.tags:
                        c = tag.code
                        name = tag.name
                        v = tag.value
                        if hasattr(v, 'value'):
                            v = v.value

                        if name == 'ExifTag' and isinstance(v, dict):
                            for sk, sv in v.items():
                                cid = sk if isinstance(sk, int) else tag_name_to_id.get(sk)
                                if cid and cid not in (0x8769, 0x8825):
                                    if hasattr(sv, 'value'):
                                        sv = sv.value
                                    try:
                                        norm_v = _normalize_exif_val(cid, sv)
                                        sub_exif[cid] = norm_v
                                    except Exception:
                                        pass
                        elif (name in ('GPSTag', 'GPSInfo') or c in (0x8825, 34853)) and isinstance(v, dict):
                            for gk, gv in v.items():
                                gid = gk if isinstance(gk, int) else gps_name_to_id.get(gk)
                                if gid is not None:
                                    if hasattr(gv, 'value'):
                                        gv = gv.value
                                    try:
                                        norm_gv = _normalize_gps_val(gid, gv)
                                        gps_exif[gid] = norm_gv
                                    except Exception:
                                        pass
                        else:
                            # Top-level IFD0 tags
                            if (c < 0x8000 or c == 0x8298 or c == 0x013B) and c not in (
                                0x8769, 0x8825, 254, 256, 257, 258, 259, 273, 277, 278, 279,
                                282, 283, 330, 513, 514, 34675, 50341, 50740
                            ):
                                try:
                                    norm_v = _normalize_exif_val(c, v)
                                    loaded_exif[c] = norm_v
                                except Exception:
                                    pass

                # Ensure DateTime is present in IFD0 if available in sub_exif, and vice-versa
                if 0x0132 not in loaded_exif and 0x9003 in sub_exif:
                    loaded_exif[0x0132] = str(sub_exif[0x9003]).strip('\x00')
                elif 0x9003 not in sub_exif and 0x0132 in loaded_exif:
                    sub_exif[0x9003] = str(loaded_exif[0x0132]).strip('\x00')
                if 0x9004 not in sub_exif:
                    sub_exif[0x9004] = sub_exif.get(0x9003) or loaded_exif.get(0x0132)

                # Ensure string tags are clean strings without internal or trailing NUL
                for target_ifd in (loaded_exif, sub_exif):
                    for k, val in list(target_ifd.items()):
                        if isinstance(val, str):
                            target_ifd[k] = val.strip('\x00')
                        elif isinstance(val, (bytes, bytearray)):
                            if k not in (0x927C, 0x9286, 0x9101, 0x9000, 0x02BC) and b'\x00' in val:
                                try:
                                    dec = val.decode('utf-8', errors='ignore').strip('\x00')
                                    if dec:
                                        target_ifd[k] = dec
                                except Exception:
                                    pass

                if 0x9000 in sub_exif:
                    v_ver = sub_exif[0x9000]
                    if isinstance(v_ver, str):
                        sub_exif[0x9000] = v_ver.encode('ascii')

                loaded_exif[0x0112] = 1  # Force upright
                try:
                    exif_bytes = loaded_exif.tobytes()
                except Exception as e:
                    logger.debug(f"tifffile tobytes: {e}")
        except Exception as e:
            logger.debug(f"tifffile native raw exif extraction: {e}")

    # 2. Fuji RAF embedded JPEG preview header
    if loaded_exif is None or len(loaded_exif) == 0:
        if ext == '.raf':
            try:
                with open(source_path, 'rb') as rf:
                    hdr = rf.read(128)
                    if hdr.startswith(b'FUJIFILMCCD-RAW'):
                        off_jpg = struct.unpack('>I', hdr[84:88])[0]
                        len_jpg = struct.unpack('>I', hdr[88:92])[0]
                        rf.seek(off_jpg)
                        jpg_bytes = rf.read(len_jpg)
                        with Image.open(io.BytesIO(jpg_bytes)) as j_img:
                            loaded_exif = j_img.getexif()
                            raw_exif = j_img.info.get('exif')
                            if raw_exif:
                                exif_bytes = raw_exif
                            elif loaded_exif:
                                loaded_exif[0x0112] = 1
                                exif_bytes = loaded_exif.tobytes()
            except Exception as e:
                logger.debug(f"RAF direct preview exif extraction: {e}")

    # 3. Direct PIL open (JPG, TIFF, PNG, DNG)
    if loaded_exif is None or len(loaded_exif) == 0:
        try:
            with Image.open(source_path) as im:
                loaded_exif = im.getexif()
                raw_exif = im.info.get('exif')
                if raw_exif:
                    exif_bytes = raw_exif
                elif loaded_exif:
                    loaded_exif[0x0112] = 1
                    try:
                        exif_bytes = loaded_exif.tobytes()
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"PIL direct exif: {e}")

    # 4. Camera RAW fallback via rawpy thumbnail preview (only if preceding failed)
    if (loaded_exif is None or len(loaded_exif) == 0) and (ext in RAW_EXTENSIONS):
        try:
            import rawpy
            with rawpy.imread(source_path) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as j_img:
                        loaded_exif = j_img.getexif()
                        raw_exif = j_img.info.get('exif')
                        if raw_exif:
                            exif_bytes = raw_exif
                        elif loaded_exif:
                            loaded_exif[0x0112] = 1
                            exif_bytes = loaded_exif.tobytes()
        except Exception as e:
            logger.debug(f"rawpy extract_thumb exif: {e}")

    # Ensure orientation is upright (tag 0x0112 = 1) across all formats and serialized bytes
    if loaded_exif is not None:
        try:
            loaded_exif[0x0112] = 1
        except Exception:
            pass

    if exif_bytes is not None:
        exif_bytes = _force_exif_orientation_upright(exif_bytes)
    elif loaded_exif:
        try:
            exif_bytes = loaded_exif.tobytes()
        except Exception:
            pass

    # Build safe TIFF extratags with signed/unsigned type safety
    tiff_tags = []
    skip_tiff_codes = {
        0x8769, 0x8825, 254, 256, 257, 258, 259, 262, 270, 273, 277, 278, 279, 284, 296, 282, 283, 305, 34675, 50341
    }
    seen_codes = set()

    def _add_tiff_tag(c, v):
        if c in skip_tiff_codes or c in seen_codes:
            return
        if c == 0x0112 or c == 274:
            v = 1
        seen_codes.add(c)
        try:
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                val_int = int(v)
                if val_int < 0:
                    tiff_tags.append((c, "i", 1, val_int, True))
                elif val_int <= 65535:
                    tiff_tags.append((c, "H", 1, val_int, True))
                else:
                    tiff_tags.append((c, "I", 1, val_int, True))
            elif hasattr(v, "numerator") and hasattr(v, "denominator"):
                num, den = int(v.numerator), int(v.denominator)
                dtype = "2i" if num < 0 or den < 0 else "2I"
                tiff_tags.append((c, dtype, 1, (num, den), True))
            elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, np.integer)) for x in v):
                dtype = "2i" if any(x < 0 for x in v) else "2I"
                tiff_tags.append((c, dtype, 1, (int(v[0]), int(v[1])), True))
            elif isinstance(v, (tuple, list)) and len(v) > 0 and all(hasattr(x, "numerator") for x in v):
                count = len(v)
                flat_vals = []
                is_signed = False
                for x in v:
                    n, d = int(x.numerator), int(x.denominator)
                    if n < 0 or d < 0:
                        is_signed = True
                    flat_vals.extend([n, d])
                dtype = "2i" if is_signed else "2I"
                tiff_tags.append((c, dtype, count, tuple(flat_vals), True))
            elif isinstance(v, str):
                tiff_tags.append((c, "s", 0, v, True))
            elif isinstance(v, (bytes, bytearray)):
                tiff_tags.append((c, "B", len(v), bytes(v), True))
        except Exception:
            pass

    if loaded_exif:
        for code, val in loaded_exif.items():
            _add_tiff_tag(code, val)
        try:
            sub = loaded_exif.get_ifd(0x8769)
            if len(sub) > 0:
                for code, val in sub.items():
                    _add_tiff_tag(code, val)
        except Exception:
            pass

    return loaded_exif, exif_bytes, tiff_tags


def save_encoded_image(rendered_data, output_path, params_dict):
    """Canonical image encoding & file writer for SpektraDarkroom.
    Saves float or integer image arrays to disk with full camera EXIF and ICC profile embedding.
    Supports JPEG, TIFF (8/16/32-bit), PNG (8/16-bit).
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_path):
        try:
            import stat
            os.chmod(output_path, stat.S_IWRITE)
        except Exception:
            pass

    from PIL import Image
    import cv2

    bit_depth = int(params_dict.get("bit_depth", 8))
    dpi_val = int(params_dict.get("dpi", 300))
    fmt_lower = str(params_dict.get("format", "jpeg")).lower()
    colorspace = params_dict.get("colorspace", "sRGB")
    source_path = params_dict.get("source_path") or params_dict.get("photo_path")

    # Standardize input array into float32 [0.0, 1.0] for color space transform / scaling
    if rendered_data.dtype == np.uint8:
        rendered_float = rendered_data.astype(np.float32) / 255.0
    elif rendered_data.dtype == np.uint16:
        rendered_float = rendered_data.astype(np.float32) / 65535.0
    else:
        rendered_float = np.clip(rendered_data.astype(np.float32), 0.0, 1.0)

    # 1. Colorspace conversion (rendered_float is in sRGB non-linear)
    if colorspace and colorspace.lower() not in ("srgb", "srgb"):
        rendered_float = convert_image_colorspace(rendered_float, colorspace)

    # 2. Extract ICC profile bytes and EXIF bundle from source image / RAW negative
    icc_bytes = get_icc_profile_bytes(colorspace)
    pil_exif, exif_bytes, tiff_tags = extract_exif_bundle(source_path)

    # 2.1 Synchronize resolution DPI and ColorSpace into EXIF
    if pil_exif is not None:
        try:
            from PIL import TiffImagePlugin
            pil_exif[282] = TiffImagePlugin.IFDRational(dpi_val, 1)
            pil_exif[283] = TiffImagePlugin.IFDRational(dpi_val, 1)
            pil_exif[296] = 2

            sub_ifd = pil_exif.get_ifd(0x8769)
            if "srgb" in str(colorspace).lower():
                sub_ifd[0xA001] = 1  # Standard sRGB
            else:
                sub_ifd[0xA001] = 65535  # Uncalibrated (Wide-gamut P3 / AdobeRGB with embedded ICC)

            new_exif_bytes = pil_exif.tobytes()
            if new_exif_bytes:
                exif_bytes = _force_exif_orientation_upright(new_exif_bytes)
        except Exception as e_sync:
            logger.debug(f"save_encoded_image EXIF tag sync: {e_sync}")

    # 3. Format-specific encoding
    if "tif" in fmt_lower:
        comp = str(params_dict.get("tiff_compression", "lzw")).lower()
        if comp in ("none", "uncompressed"):
            comp_val = None
        elif comp in ("zip", "deflate", "zlib"):
            comp_val = "deflate"
        else:
            comp_val = "lzw"

        extratags = []
        if icc_bytes:
            extratags.append((34675, 'B', len(icc_bytes), icc_bytes, False))
        if tiff_tags:
            extratags.extend(tiff_tags)

        def _safe_tifffile_write(path, data):
            import tifffile
            for attempt in [comp_val, "deflate", None]:
                try:
                    tifffile.imwrite(path, data, compression=attempt, resolution=(dpi_val, dpi_val, 'INCH'), photometric='rgb', extratags=extratags if extratags else None)
                    return
                except Exception:
                    continue
            try:
                tifffile.imwrite(path, data, resolution=(dpi_val, dpi_val, 'INCH'), photometric='rgb', extratags=extratags if extratags else None)
            except Exception:
                im = Image.fromarray(data if data.dtype == np.uint8 else (data / 256).astype(np.uint8))
                save_kw = {'dpi': (dpi_val, dpi_val)}
                if icc_bytes:
                    save_kw['icc_profile'] = icc_bytes
                if exif_bytes:
                    save_kw['exif'] = exif_bytes
                im.save(path, 'TIFF', **save_kw)

        if bit_depth == 32:
            out_f32 = rendered_float.astype(np.float32)
            _safe_tifffile_write(output_path, out_f32)
        elif bit_depth == 16:
            out_u16 = (np.clip(rendered_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
            _safe_tifffile_write(output_path, out_u16)
        else:
            out_u8 = (np.clip(rendered_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            _safe_tifffile_write(output_path, out_u8)

    elif "png" in fmt_lower:
        png_level = int(params_dict.get("png_level", 6))
        if bit_depth == 16:
            out_u16 = (np.clip(rendered_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
            bgr_u16 = cv2.cvtColor(out_u16, cv2.COLOR_RGB2BGR)
            ok, enc_buf = cv2.imencode('.png', bgr_u16, [cv2.IMWRITE_PNG_COMPRESSION, png_level])
            if ok:
                pdata = enc_buf.tobytes()
                # Inject EXIF and ICC into 16-bit PNG in memory
                try:
                    if pdata.startswith(b'\x89PNG\r\n\x1a\n'):
                        import struct, zlib
                        idx = 8
                        ihdr_len = struct.unpack('>I', pdata[idx:idx+4])[0]
                        insert_pos = idx + 8 + ihdr_len + 4
                        injected_chunks = []
                        if icc_bytes:
                            comp_icc = zlib.compress(icc_bytes)
                            cdata = b'ICC Profile\x00\x00' + comp_icc
                            injected_chunks.append(struct.pack('>I', len(cdata)) + b'iCCP' + cdata + struct.pack('>I', zlib.crc32(b'iCCP' + cdata) & 0xffffffff))
                        if exif_bytes:
                            clean_exif = exif_bytes[6:] if exif_bytes.startswith(b'Exif\x00\x00') else exif_bytes
                            injected_chunks.append(struct.pack('>I', len(clean_exif)) + b'eXIf' + clean_exif + struct.pack('>I', zlib.crc32(b'eXIf' + clean_exif) & 0xffffffff))
                        if injected_chunks:
                            pdata = pdata[:insert_pos] + b''.join(injected_chunks) + pdata[insert_pos:]
                except Exception as ex_png:
                    print(f"[SpektraEngine] Warning: PNG 16-bit metadata injection failed: {ex_png}")
                with open(output_path, 'wb') as pf:
                    pf.write(pdata)
        else:
            out_u8 = (np.clip(rendered_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            im = Image.fromarray(out_u8, mode='RGB')
            save_kw = {'compress_level': png_level, 'dpi': (dpi_val, dpi_val)}
            if icc_bytes:
                save_kw['icc_profile'] = icc_bytes
            if exif_bytes:
                save_kw['exif'] = exif_bytes
            im.save(output_path, 'PNG', **save_kw)

    else:  # JPEG
        q_scale = int(params_dict.get("quality", 9))
        q_val = 50 + q_scale * 5 if q_scale <= 10 else 95
        q_val = max(10, min(100, q_val))

        sub_opt = str(params_dict.get("subsampling", "4:4:4"))
        sub_val = 0 if sub_opt == "4:4:4" else (1 if sub_opt == "4:2:2" else 2)
        prog_val = bool(params_dict.get("progressive", True))

        out_u8 = (np.clip(rendered_float, 0.0, 1.0) * 255.0).astype(np.uint8)
        scale_pct = int(params_dict.get("scale_pct", 100))
        if 0 < scale_pct < 100:
            target_w = max(1, int(out_u8.shape[1] * scale_pct / 100.0))
            target_h = max(1, int(out_u8.shape[0] * scale_pct / 100.0))
            out_u8 = cv2.resize(out_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)

        im = Image.fromarray(out_u8, mode='RGB')
        save_kw = {
            'quality': q_val,
            'subsampling': sub_val,
            'progressive': prog_val,
            'dpi': (dpi_val, dpi_val)
        }
        if icc_bytes:
            save_kw['icc_profile'] = icc_bytes
        if exif_bytes:
            save_kw['exif'] = exif_bytes
        try:
            im.save(output_path, 'JPEG', **save_kw)
        except OSError:
            if save_kw.get('progressive'):
                save_kw['progressive'] = False
                im.save(output_path, 'JPEG', **save_kw)
            else:
                raise

    return True


class SpektraEngine:
    def __init__(self, resources_dir=None):
        if resources_dir is None:
            from .path_utils import get_resource_dir
            self.resources_dir = os.path.abspath(get_resource_dir())
        else:
            self.resources_dir = os.path.abspath(resources_dir)
        self.profiles_dir = os.path.join(self.resources_dir, "profiles")
        self.filters_file = os.path.join(self.resources_dir, "neutral_print_filters.json")
        
        # Load neutral print filters table
        self.neutral_filters = {}
        if os.path.exists(self.filters_file):
            try:
                with open(self.filters_file, "r", encoding="utf-8") as f:
                    self.neutral_filters = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load neutral_print_filters.json: {e}")

        # Multi-photo library session
        self.session_photos = []
        self.active_photo_id = None

        # Active image buffers
        self.current_file_path = None
        self.raw_full = None
        self.raw_preview = None   # float32 [H, W, 3] in [0, 1] (high-res for GPU texture)
        self.original_meta = {}

        # Performance caches
        self._lut_cache = {}      # (film, paper, size) -> np.ndarray [N, N, N, 3] float32
        self._profile_cache = {}
        self._cached_sim = None
        self._cached_config = None
        self._render_lock = threading.Lock()

    def _load_profile(self, stock_id):
        if not stock_id or stock_id == "none":
            return None
        if stock_id not in self._profile_cache:
            custom_path = os.path.join(self.profiles_dir, f"{stock_id}.json")
            if os.path.exists(custom_path):
                from spektrafilm.profiles.io import profile_from_dict
                with open(custom_path, "r", encoding="utf-8") as fp:
                    self._profile_cache[stock_id] = profile_from_dict(json.load(fp))
            else:
                self._profile_cache[stock_id] = load_processed_profile(stock_id)
        return self._profile_cache[stock_id]

    BUILTIN_FILM_IDS = {
        "none", "kodak_portra_400", "kodak_portra_160", "kodak_portra_800",
        "kodak_portra_800_push1", "kodak_portra_800_push2", "kodak_gold_200",
        "kodak_ektar_100", "kodak_ultramax_400", "fujifilm_pro_400h", "fujifilm_c200",
        "fujifilm_xtra_400", "kodak_vision3_500t", "kodak_vision3_250d",
        "kodak_vision3_50d", "kodak_vision3_200t", "kodak_verita_200d",
        "fujifilm_velvia_100", "fujifilm_provia_100f", "kodak_ektachrome_100",
        "kodak_kodachrome_64"
    }

    BUILTIN_PAPER_IDS = {
        "none", "kodak_2383", "kodak_2393", "kodak_portra_endura", "kodak_ultra_endura",
        "kodak_supra_endura", "kodak_endura_premier", "kodak_ektacolor_edge",
        "fujifilm_crystal_archive_typeii"
    }

    def get_film_stocks(self):
        stocks = [
            {
                "category": "数码原生 (Bypass)",
                "stocks": [
                    {
                        "id": "none",
                        "name": "无",
                        "badge": "原生",
                        "desc": "不加载任何胶卷乳剂模拟，直接呈现数码传感器原生原始色彩与反差。"
                    }
                ]
            },
            {
                "category": "顶流色彩负片 (Color Negative)",
                "stocks": [
                    {
                        "id": "kodak_portra_400",
                        "name": "Kodak Portra 400",
                        "badge": "人像之王",
                        "desc": "最受全球摄影师推崇的胶片，细腻温润的中间调与真实自然的肤色，高光宽容度极其优秀。"
                    },
                    {
                        "id": "kodak_portra_160",
                        "name": "Kodak Portra 160",
                        "badge": "纯净柔和",
                        "desc": "超细腻银盐微粒，淡雅清爽的高级杂志影调与自然光质感。"
                    },
                    {
                        "id": "kodak_portra_800",
                        "name": "Kodak Portra 800",
                        "badge": "弱光人像",
                        "desc": "高感人像负片，弱光环境与柔和暗部过渡，色彩表现极具温度。"
                    },
                    {
                        "id": "kodak_portra_800_push1",
                        "name": "Kodak Portra 800 (Push +1)",
                        "badge": "迫冲1档",
                        "desc": "迫冲 1 档显影，反差与微粒质感适度提升，戏剧化影调。"
                    },
                    {
                        "id": "kodak_portra_800_push2",
                        "name": "Kodak Portra 800 (Push +2)",
                        "badge": "迫冲2档",
                        "desc": "迫冲 2 档显影，戏剧性强烈反差与粗粝微粒质感，极具表现力。"
                    },
                    {
                        "id": "kodak_gold_200",
                        "name": "Kodak Gold 200",
                        "badge": "怀旧金黄",
                        "desc": "夏日阳光感、温暖的黄色调与浓郁复古胶片氛围，扫街与纪实首选。"
                    },
                    {
                        "id": "kodak_ektar_100",
                        "name": "Kodak Ektar 100",
                        "badge": "极致微反差",
                        "desc": "极高色彩饱和度与微反差锐度，风光摄影与大光比静物的不二之选。"
                    },
                    {
                        "id": "kodak_ultramax_400",
                        "name": "Kodak UltraMax 400",
                        "badge": "浓郁街拍",
                        "desc": "高感光度下生动鲜明的原色反差，经典的平民纪实街头风范。"
                    },
                    {
                        "id": "fujifilm_pro_400h",
                        "name": "Fujifilm Pro 400H",
                        "badge": "日系空气感",
                        "desc": "经典日系色调的代表作，标志性的青绿薄荷暗部与通透粉嫩的肤色。"
                    },
                    {
                        "id": "fujifilm_c200",
                        "name": "Fujifilm C200",
                        "badge": "日系清新",
                        "desc": "柔和自然的蓝绿过渡与轻快的层次感，极具生活烟火气。"
                    },
                    {
                        "id": "fujifilm_xtra_400",
                        "name": "Fujifilm Superia X-TRA 400",
                        "badge": "浓烈青红",
                        "desc": "第四感光层加持的高宽容度，鲜明的洋红与青绿冷暖反差。"
                    },
                    {
                        "id": "kodak_verita_200d",
                        "name": "Kodak Verita 200D",
                        "badge": "日光配方",
                        "desc": "柯达实验性日光胶卷配方，精准的自然光谱响应与饱满色彩层次。"
                    }
                ]
            },
            {
                "category": "电影工业胶片 (Cinema Film / Vision3)",
                "stocks": [
                    {
                        "id": "kodak_vision3_500t",
                        "name": "Kodak Vision3 500T",
                        "badge": "CineStill 800T原型",
                        "desc": "好莱坞钨丝灯电影卷，深邃冷调与夜景霓虹灯下标志性的迷人红光晕（Halation）。"
                    },
                    {
                        "id": "kodak_vision3_250d",
                        "name": "Kodak Vision3 250D",
                        "badge": "好莱坞日光卷",
                        "desc": "现代电影工业标准日光底片，厚实耐看的影调，无与伦比的高光保留能力。"
                    },
                    {
                        "id": "kodak_vision3_200t",
                        "name": "Kodak Vision3 200T",
                        "badge": "钨丝室内卷",
                        "desc": "灯光型电影底片，专为室内及夜景人工光源设计，平衡耐看。"
                    },
                    {
                        "id": "kodak_vision3_50d",
                        "name": "Kodak Vision3 50D",
                        "badge": "极低噪点",
                        "desc": "世界上最细腻的彩色负片之一，纯净的电影胶片质感与细腻层次。"
                    }
                ]
            },
            {
                "category": "传奇正片/反转片 (Color Reversal / Slide)",
                "stocks": [
                    {
                        "id": "fujifilm_velvia_100",
                        "name": "Fujifilm Velvia 100",
                        "badge": "风光传奇",
                        "desc": "色彩浓郁欲滴、极高对比度与深沉黑位，大自然壮阔风光的代名词。"
                    },
                    {
                        "id": "fujifilm_provia_100f",
                        "name": "Fujifilm Provia 100F",
                        "badge": "真实中性",
                        "desc": "业界标杆级的精准中性色彩还原与平滑过渡。"
                    },
                    {
                        "id": "kodak_ektachrome_100",
                        "name": "Kodak Ektachrome E100",
                        "badge": "通透冷艳",
                        "desc": "清澈通透的蓝色基调与极高解像力，现代反转片的杰作。"
                    },
                    {
                        "id": "kodak_kodachrome_64",
                        "name": "Kodak Kodachrome 64",
                        "badge": "染印传奇",
                        "desc": "柯达一代传奇染印法反转片配方，浓烈深沉的红色调与世纪经典纪实色彩。"
                    }
                ]
            }
        ]
        # Dynamically discover user-imported custom film profiles
        custom_films = []
        if os.path.exists(self.profiles_dir):
            import json
            for f in sorted(os.listdir(self.profiles_dir)):
                if f.endswith(".json"):
                    sid = f[:-5]
                    if sid not in self.BUILTIN_FILM_IDS and sid not in self.BUILTIN_PAPER_IDS:
                        try:
                            with open(os.path.join(self.profiles_dir, f), "r", encoding="utf-8") as fp:
                                meta = json.load(fp)
                            info_obj = meta.get("info", {})
                            if meta.get("type") != "paper" and "paper" not in sid.lower() and info_obj.get("support") != "paper":
                                channel_model = info_obj.get("channel_model", "color")
                                film_type = info_obj.get("type", "negative")
                                badge = meta.get("badge") or info_obj.get("badge")
                                if not badge:
                                    if "aerochrome" in sid or "infrared" in sid:
                                        badge = "红外假色"
                                    elif "cinestill" in sid or "eterna" in sid or "double_x" in sid or info_obj.get("use") == "cine":
                                        badge = "电影底片"
                                    elif "phoenix" in sid or "lomochrome" in sid or "lomography" in sid:
                                        badge = "实验胶卷"
                                    elif channel_model == "bw":
                                        badge = "黑白银盐"
                                    elif film_type == "positive":
                                        badge = "正片反转"
                                    else:
                                        badge = "经典彩负"

                                name_str = meta.get("name") or info_obj.get("name") or sid.replace("_", " ").title()
                                desc_str = meta.get("desc") or info_obj.get("desc") or f"{name_str} 81波段全光谱物理档案"
                                custom_films.append({
                                    "id": sid,
                                    "name": name_str,
                                    "badge": badge,
                                    "desc": desc_str,
                                    "channel_model": channel_model,
                                    "type": film_type,
                                    "use": info_obj.get("use", "still"),
                                    "is_custom": True
                                })
                        except Exception:
                            pass
        if custom_films:
            stocks.append({
                "category": "物理测定胶卷扩充库 (Fotufilm & Custom)",
                "stocks": custom_films
            })

        # Ensure all stock dicts have profile metadata (channel_model, type, use)
        if not hasattr(self, "_stock_meta_cache"):
            self._stock_meta_cache = {}
        for group in stocks:
            for item in group.get("stocks", []):
                sid = item.get("id")
                if sid == "none":
                    item["channel_model"] = "color"
                    item["type"] = "bypass"
                    item["use"] = "bypass"
                    continue
                if sid in self._stock_meta_cache:
                    cached_meta = self._stock_meta_cache[sid]
                    item["channel_model"] = cached_meta.get("channel_model", "color")
                    item["type"] = cached_meta.get("type", "negative")
                    item["use"] = cached_meta.get("use", "still")
                else:
                    p_path = os.path.join(self.profiles_dir, f"{sid}.json")
                    if os.path.exists(p_path):
                        try:
                            with open(p_path, "r", encoding="utf-8") as fp:
                                meta = json.load(fp)
                            inf = meta.get("info", {})
                            c_model = inf.get("channel_model", "color")
                            f_type = inf.get("type", "negative")
                            f_use = inf.get("use", "still")
                            item["channel_model"] = c_model
                            item["type"] = f_type
                            item["use"] = f_use
                            self._stock_meta_cache[sid] = {
                                "channel_model": c_model,
                                "type": f_type,
                                "use": f_use
                            }
                        except Exception:
                            item.setdefault("channel_model", "color")
                            item.setdefault("type", "negative")
                            item.setdefault("use", "still")

        return stocks

    def get_paper_stocks(self):
        papers = [
            {
                "id": "none",
                "name": "数码扫描",
                "badge": "数码直扫",
                "desc": "跳过相纸印相，呈现底片直接数码扫描的通透细节与宽容度。"
            },
            {
                "id": "kodak_2383",
                "name": "Kodak Vision 2383",
                "badge": "好莱坞标准",
                "desc": "电影放映拷贝标准片，带来深沉厚重的黑位（D-max）与经典的青橙分色立体感。"
            },
            {
                "id": "kodak_2393",
                "name": "Kodak Vision 2393",
                "badge": "高密黑阶",
                "desc": "专业电影放映正片，拥有极高的最大光学密度与纯净高光分离感。"
            },
            {
                "id": "kodak_portra_endura",
                "name": "Kodak Portra Endura",
                "badge": "人像展览级",
                "desc": "暗房人像定制相纸，高光平滑过渡，呈现极其优雅柔和的皮肤质感。"
            },
            {
                "id": "kodak_ultra_endura",
                "name": "Kodak Ultra Endura",
                "badge": "通透锐利",
                "desc": "更高的反差和色彩浓度，适合现代风光与建筑摄影的有力呈现。"
            },
            {
                "id": "kodak_supra_endura",
                "name": "Kodak Supra Endura",
                "badge": "均衡旗舰",
                "desc": "全能型专业暗房相纸，动态范围广阔，色彩还原生动饱和。"
            },
            {
                "id": "kodak_endura_premier",
                "name": "Kodak Endura Premier",
                "badge": "数码暗房顶峰",
                "desc": "柯达顶级数码暗房相纸，扩展色彩域与高光洁白度。"
            },
            {
                "id": "kodak_ektacolor_edge",
                "name": "Kodak Ektacolor Edge",
                "badge": "平民经典",
                "desc": "柯达广受欢迎的日常彩色冲印相纸，温暖平实的生活感色彩。"
            },
            {
                "id": "fujifilm_crystal_archive_typeii",
                "name": "Fujifilm Crystal Archive II",
                "badge": "纯净高光",
                "desc": "富士纯白高光表现，色彩鲜艳明快，层次丰富持久。"
            }
        ]
        # Dynamically discover user-imported custom paper profiles
        if os.path.exists(self.profiles_dir):
            import json
            for f in sorted(os.listdir(self.profiles_dir)):
                if f.endswith(".json"):
                    sid = f[:-5]
                    if sid not in self.BUILTIN_FILM_IDS and sid not in self.BUILTIN_PAPER_IDS:
                        try:
                            with open(os.path.join(self.profiles_dir, f), "r", encoding="utf-8") as fp:
                                meta = json.load(fp)
                            if meta.get("type") == "paper" or "paper" in sid.lower():
                                papers.append({
                                    "id": sid,
                                    "name": meta.get("name", sid),
                                    "badge": "自定义相纸",
                                    "desc": meta.get("desc") or (meta.get("info", {}).get("desc", "")) or "用户自定义相纸配置",
                                    "is_custom": True
                                })
                        except Exception:
                            pass
        return papers

    def get_neutral_filter(self, paper_stock, film_stock):
        if paper_stock in self.neutral_filters:
            paper_dict = self.neutral_filters[paper_stock]
            for illuminant_key in ["TH-KG3", "tungsten", "default"]:
                if illuminant_key in paper_dict and film_stock in paper_dict[illuminant_key]:
                    return list(paper_dict[illuminant_key][film_stock])

        # If positive / reversal slide film or monochrome, return neutral baseline
        if self.is_positive_film(film_stock):
            return [0.0, 50.0, 50.0]
        prof = self._load_profile(film_stock)
        if prof and hasattr(prof, 'info') and getattr(prof.info, 'channel_model', 'color') == 'bw':
            return [0.0, 0.0, 0.0]

        # Automatic on-the-fly solver for arbitrary custom negative film stocks
        try:
            from scipy.optimize import root
            from spektrafilm.model.illuminants import standard_illuminant
            from spektrafilm.model.color_filters import color_enlarger
            from spektrafilm.utils.conversions import density_to_light
            from spektrafilm.model.develop import compute_density_spectral

            paper_prof = self._load_profile(paper_stock)
            if prof and paper_prof:
                paper_sens = 10 ** paper_prof.data.log_sensitivity
                lamp = standard_illuminant("TH-KG3")

                ref_film = self._load_profile("kodak_portra_400")
                if ref_film:
                    ref_idx0 = int(np.argmin(np.abs(ref_film.data.log_exposure)))
                    ref_cmy_mid = ref_film.data.density_curves[ref_idx0:ref_idx0+1][None, :, :]
                    ref_ds = compute_density_spectral(ref_film.data.channel_density, ref_cmy_mid, base_density=ref_film.data.base_density)
                    cal_filt = self.get_neutral_filter(paper_stock, "kodak_portra_400")
                    filtered_lamp = color_enlarger(lamp, cal_filt)
                    light = density_to_light(ref_ds, filtered_lamp)
                    raw_p = np.einsum('ijk, kl->ijl', light, paper_sens)[0, 0]
                    target_g, target_b = (raw_p[1] / raw_p[0], raw_p[2] / raw_p[0])

                    idx_0 = int(np.argmin(np.abs(prof.data.log_exposure)))
                    cmy_mid = prof.data.density_curves[idx_0:idx_0+1][None, :, :]
                    ds = compute_density_spectral(prof.data.channel_density, cmy_mid, base_density=prof.data.base_density)

                    def obj(my):
                        fl = color_enlarger(lamp, [0.0, my[0], my[1]])
                        li = density_to_light(ds, fl)
                        rp = np.einsum('ijk, kl->ijl', li, paper_sens)[0, 0]
                        return [rp[1] / rp[0] - target_g, rp[2] / rp[0] - target_b]

                    sol = root(obj, [60.0, 60.0], method='hybr')
                    if sol.success and np.all(np.isfinite(sol.x)):
                        res = [0.0, float(sol.x[0]), float(sol.x[1])]
                        if paper_stock not in self.neutral_filters:
                            self.neutral_filters[paper_stock] = {}
                        if "TH-KG3" not in self.neutral_filters[paper_stock]:
                            self.neutral_filters[paper_stock]["TH-KG3"] = {}
                        self.neutral_filters[paper_stock]["TH-KG3"][film_stock] = res
                        return res
        except Exception as e:
            print(f"[SpektraEngine] On-the-fly filter solve fallback: {e}")

        return [0.0, 60.0, 60.0]

    def is_positive_film(self, film_stock: str) -> bool:
        """Returns True if the film stock is a positive/reversal (slide) film."""
        if not film_stock or film_stock == "none":
            return False
        prof = self._load_profile(film_stock)
        if not prof:
            return False
        return bool(
            (hasattr(prof, 'is_positive') and prof.is_positive) or
            (hasattr(prof, 'info') and getattr(prof.info, 'type', None) == 'positive')
        )

    def get_film_halation_preset(self, film_stock):
        """Query official spektrafilm halation preset from film profile's (use, antihalation) tags."""
        if not film_stock or film_stock == "none":
            return {"sigma_h": 65.0, "strength": (0.015, 0.005, 0.0), "antihalation": "none", "use": "still"}
        try:
            prof = self._load_profile(film_stock)
            use = getattr(prof.info, "use", "still") if prof and hasattr(prof, "info") else "still"
            antihalation = getattr(prof.info, "antihalation", "strong") if prof and hasattr(prof, "info") else "strong"

            # Check for physical film_base extension (Phase 4)
            if prof and hasattr(prof, "data") and hasattr(prof.data, "physical_extensions"):
                phys_ext = prof.data.physical_extensions or {}
                if isinstance(phys_ext, dict):
                    fb = phys_ext.get("halation") or phys_ext.get("film_base")
                    if fb and isinstance(fb, dict):
                        thickness_mm = float(fb.get("base_thickness_mm", fb.get("thickness_mm", 0.13)))
                        thickness_um = thickness_mm * 1000.0
                        indices = fb.get("record_refractive_indices")
                        if not indices or len(indices) != 3:
                            n_val = float(fb.get("refractive_index", 1.48))
                            indices = [n_val * 0.996, n_val, n_val * 1.006]
                        radii_um = [2.0 * thickness_um * np.tan(np.arcsin(1.0 / max(float(n), 1.01))) for n in indices]
                        sigma_h = radii_um[0] / np.sqrt(2.0)
                        
                        if "strength" in fb and isinstance(fb["strength"], (list, tuple)):
                            raw_st = np.array(fb["strength"], dtype=np.float64)
                            if np.max(raw_st) > 0:
                                strength = tuple(float(x) for x in raw_st)
                            else:
                                strength = (0.0055, 0.001, 0.0)
                        else:
                            ah = str(fb.get("antihalation", antihalation)).lower()
                            if ah in ("no", "none"):
                                strength = (0.28, 0.07, 0.015)
                            elif ah == "weak":
                                strength = (0.08, 0.02, 0.005)
                            else:
                                strength = (0.015, 0.003, 0.0)
                        return {
                            "sigma_h": float(sigma_h),
                            "strength": strength,
                            "antihalation": fb.get("antihalation", antihalation),
                            "use": use
                        }

            from spektrafilm.runtime.params_builder import _HALATION_PRESETS
            preset = _HALATION_PRESETS.get((use, antihalation), {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.015, 0.005, 0.0)})
            return {
                "sigma_h": float(preset['sigma_h'][0]),
                "strength": tuple(float(x) for x in preset['strength']),
                "antihalation": antihalation,
                "use": use
            }
        except Exception:
            return {"sigma_h": 65.0, "strength": (0.015, 0.005, 0.0), "antihalation": "strong", "use": "still"}

    def get_film_grain_preset(self, film_stock):
        """Extract physical crystal grain parameters (size_mm, strength, layer_weights, density_profile)
        from film profile JSON, or provide physically-calibrated fallbacks.
        """
        if not hasattr(self, "_grain_preset_cache"):
            self._grain_preset_cache = {}

        if not film_stock or film_stock == "none":
            return {
                "size_mm": 0.005,
                "strength": 0.010,
                "layer_weights": [0.70, 1.0, 1.35],
                "density_profile": None,
                "density_law": 2
            }

        if film_stock in self._grain_preset_cache:
            return self._grain_preset_cache[film_stock]

        preset = None
        custom_path = os.path.join(self.profiles_dir, f"{film_stock}.json")
        if os.path.exists(custom_path):
            try:
                with open(custom_path, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    cg = None
                    if "data" in d and isinstance(d["data"], dict) and "physical_extensions" in d["data"]:
                        cg = d["data"]["physical_extensions"].get("crystal_grain")
                    elif "physical_extensions" in d and isinstance(d["physical_extensions"], dict):
                        cg = d["physical_extensions"].get("crystal_grain")

                    if cg and isinstance(cg, dict):
                        preset = {
                            "size_mm": float(cg.get("size_mm", 0.005)),
                            "strength": float(cg.get("strength", 0.010)),
                            "layer_weights": [float(x) for x in cg.get("layer_weights", [0.70, 1.0, 1.35])],
                            "density_profile": cg.get("density_profile"),
                            "density_law": int(cg.get("density_law", 2))
                        }
            except Exception as e:
                print(f"[SpektraGrain] 提取胶片颗粒配置异常: {e}")

        # Intelligent physical fallback if not in profile
        if preset is None:
            stock_lower = film_stock.lower()
            is_bw = any(k in stock_lower for k in ("mono", "b&w", "bw", "tmax", "tri_x", "hp5", "delta", "pan_f", "acros", "double_x", "infrared"))
            is_tungsten = any(k in stock_lower for k in ("800t", "500t", "200t", "tungsten"))
            is_fast = any(k in stock_lower for k in ("400", "800", "1600", "3200"))
            is_slow = any(k in stock_lower for k in ("25", "50", "64", "100", "125", "160"))

            if is_bw:
                layer_weights = [1.0, 1.0, 1.0]
            elif is_tungsten:
                layer_weights = [0.90, 1.0, 2.71]
            else:
                layer_weights = [0.70, 1.0, 1.35]

            if "3200" in stock_lower:
                size_mm = 0.008
                strength = 0.019
            elif "tri_x" in stock_lower or "hp5" in stock_lower:
                size_mm = 0.008
                strength = 0.017
            elif is_fast:
                size_mm = 0.005
                strength = 0.011
            elif is_slow:
                size_mm = 0.004
                strength = 0.006
            else:
                size_mm = 0.005
                strength = 0.010

            preset = {
                "size_mm": size_mm,
                "strength": strength,
                "layer_weights": layer_weights,
                "density_profile": None,
                "density_law": 2
            }

        self._grain_preset_cache[film_stock] = preset
        return preset

    @staticmethod
    def _postprocess_positive_lut(lut_3d: np.ndarray, film_stock: str) -> np.ndarray:
        """
        Calibrate and shape 3D LUT for positive reversal slide film:
        Stage 1. Full-Spectrum Neutral-Curve Reversal Balance:
                 Maps raw non-linear dye transmittances along the neutral diagonal
                 to strictly balanced Rec.709 neutral luminance Y(t), eliminating
                 raw dye color casts (e.g. Velvia 50 red bias) while keeping neutral grays neutral.
        Stage 2. Clear Base D-min White Normalization:
                 Calibrates clear film base transmittance (1.0, 1.0, 1.0) to display white (1.0).
        Stage 3. Midtone-Preserving Slide S-Curve:
                 Anchored at 18% reflectance mid-gray (pivot p = 0.18, gamma g = 1.15)
                 to avoid midtone darkening and open shadow details.
        Stage 4. Perceptual Oklab Chroma Expansion:
                 Expands chroma in Oklab space without altering perceptual lightness,
                 yielding vibrant, punchy slide film colors (Fotufilm reference match).
        """
        lut_size = lut_3d.shape[0]
        # 1. Convert from non-linear sRGB back to linear transmittance
        lin = np.where(lut_3d <= 0.04045, lut_3d / 12.92, ((lut_3d + 0.055) / 1.055) ** 2.4).astype(np.float32)

        # 2. Stage 1: Full-Spectrum Neutral-Curve Reversal Balance
        diag = np.array([lin[i, i, i] for i in range(lut_size)], dtype=np.float32)
        diag_mono = np.zeros_like(diag)
        for c in range(3):
            diag_mono[:, c] = np.maximum.accumulate(diag[:, c])

        for i in range(lut_size):
            for c in range(3):
                if lin[i, i, i, c] < diag_mono[i, c]:
                    lin[i, i, i, c] = diag_mono[i, c]

        target_neutral = np.maximum.accumulate(0.2126 * diag_mono[:, 0] + 0.7152 * diag_mono[:, 1] + 0.0722 * diag_mono[:, 2])

        lin_balanced = np.zeros_like(lin)
        for c in range(3):
            u_axis, u_idx = np.unique(diag_mono[:, c], return_index=True)
            u_target = target_neutral[u_idx]
            if u_axis[-1] < diag_mono[-1, c]:
                u_axis = np.append(u_axis, diag_mono[-1, c])
                u_target = np.append(u_target, target_neutral[-1])
            else:
                u_target[-1] = target_neutral[-1]
            lin_balanced[..., c] = np.interp(lin[..., c], u_axis, u_target)

        # 3. Stage 2: Clear Base D-min White Normalization
        w_max = float(np.max(lin_balanced[lut_size - 1, lut_size - 1, lut_size - 1]))
        if w_max > 1e-4:
            v = np.clip(lin_balanced / w_max, 0.0, 1.0)
        else:
            v = lin_balanced

        # 4. Stage 3: Midtone-Preserving Slide S-Curve (pivot p = 0.18, gamma g = 1.15)
        p = 0.18
        g = 1.15
        factor = (p / (1.0 - p)) ** (g - 1.0)
        denom = (v ** g) + factor * ((1.0 - v) ** g) + 1e-12
        mapped_lin = (v ** g) / denom

        # 5. Stage 4: Perceptual Oklab Chroma Expansion
        def linear_to_oklab(lin_rgb):
            M1 = np.array([
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]
            ], dtype=np.float32)
            lms = np.clip(lin_rgb @ M1.T, 0.0, None)
            lms_cbrt = np.cbrt(lms)
            M2 = np.array([
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]
            ], dtype=np.float32)
            return lms_cbrt @ M2.T

        def oklab_to_linear(lab):
            M2_inv = np.array([
                [1.0, 0.3963377774, 0.2158037573],
                [1.0, -0.1055613458, -0.0638541728],
                [1.0, -0.0894841775, -1.2914855480]
            ], dtype=np.float32)
            lms_cbrt = lab @ M2_inv.T
            lms = lms_cbrt ** 3
            M1_inv = np.array([
                [4.0767416621, -3.3077115913, 0.2309699292],
                [-1.2684380046, 2.6097574011, -0.3413193965],
                [-0.0041960863, -0.7034186147, 1.7076147010]
            ], dtype=np.float32)
            return np.clip(lms @ M1_inv.T, 0.0, 1.0)

        fs_low = str(film_stock).lower()
        if "velvia" in fs_low or "aerochrome" in fs_low or "infrared" in fs_low:
            sat_boost = 1.35
        else:
            sat_boost = 1.25

        lab = linear_to_oklab(mapped_lin)
        lab[..., 1] *= sat_boost
        lab[..., 2] *= sat_boost
        mapped_lin = oklab_to_linear(lab)

        # 6. Convert back to display sRGB
        lut_out = np.where(
            mapped_lin <= 0.0031308,
            mapped_lin * 12.92,
            1.055 * (np.maximum(mapped_lin, 0.0) ** (1.0 / 2.4)) - 0.055
        ).astype(np.float32)
        return np.clip(lut_out, 0.0, 1.0)

    def get_3d_lut(self, film_stock="kodak_portra_400", paper_stock="kodak_2383", lut_size=33, params_dict=None, print_mode=None):
        """Build and cache a 3D LUT (N x N x N x 3 float32) for the film + paper combination."""
        params_dict = params_dict or {}
        if print_mode is None:
            print_mode = str(params_dict.get("print_mode", "optical")).lower()
        else:
            print_mode = str(print_mode).lower()
        if print_mode not in ("optical", "scan"):
            print_mode = "optical"

        illuminant = str(params_dict.get("enlarger_illuminant", "TH-KG3"))
        dir_amt = round(float(params_dict.get("dir_amount", 1.0)), 2)
        dir_intl = round(float(params_dict.get("dir_interlayer", 1.0)), 2)
        dir_same = round(float(params_dict.get("dir_samelayer", 1.0)), 2)
        morph_gamma = round(float(params_dict.get("morph_gamma", 1.0)), 2)
        dev_exh = round(float(params_dict.get("developer_exhaustion", 0.0)), 2)

        is_default_params = (
            illuminant in ("TH-KG3", "lamp") and
            abs(dir_amt - 1.0) < 0.01 and abs(dir_intl - 1.0) < 0.01 and abs(dir_same - 1.0) < 0.01 and
            abs(morph_gamma - 1.0) < 0.01 and abs(dev_exh) < 0.01
        )

        eff_paper = "none" if self.is_positive_film(film_stock) else ("scan" if print_mode == "scan" else paper_stock)
        cache_key = (film_stock, eff_paper, lut_size, illuminant, dir_amt, dir_intl, dir_same, morph_gamma, dev_exh)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]

        # 1. Shaper lattice coordinates t in [0.0, 1.0] -> Scene-linear HDR ProPhoto [0.0, 8.0]
        lin_t = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        lin_x = shaper_inverse(lin_t)
        b, g, r = np.meshgrid(lin_x, lin_x, lin_x, indexing='ij')
        lattice = np.stack([r, g, b], axis=-1).astype(np.float32)

        # 2. Bypass mode: 胶卷为 "none" 时，生成从 Linear ProPhoto RGB [0.0, 8.0] -> Display sRGB 的平滑高光胶片色调转换表
        if film_stock == "none":
            identity_lut = prophoto_to_srgb_filmic(lattice)
            self._lut_cache[cache_key] = identity_lut
            return identity_lut

        cache_dir = os.path.join(self.resources_dir, ".lut_cache")
        if is_default_params:
            cache_file = os.path.join(cache_dir, f"v4_shaper__{film_stock}__{eff_paper}__{lut_size}.npy")
            if os.path.exists(cache_file):
                try:
                    lut_3d = np.load(cache_file)
                    self._lut_cache[cache_key] = lut_3d
                    return lut_3d
                except Exception:
                    pass

        print(f"[SpektraEngine] 正在为 GPU 生成物理对数 3D LUT: {film_stock} + {eff_paper} (模式: {print_mode}, 光源: {illuminant}, DIR: {dir_intl}, 尺寸: {lut_size})...")
        t0 = time.time()
        
        flat_input = lattice.reshape(-1, 1, 3)

        params = init_params()
        params.film = self._load_profile(film_stock)

        # Check if positive (slide/reversal), B&W (monochrome), or negative film
        is_pos = (params.film and hasattr(params.film, 'is_positive') and params.film.is_positive) or \
                 (params.film and hasattr(params.film, 'info') and getattr(params.film.info, 'type', None) == 'positive')
        is_bw = (params.film and hasattr(params.film, 'info') and getattr(params.film.info, 'channel_model', None) == 'bw')

        if is_pos:
            # Positive slide film: inherently positive, no photographic print paper
            params.io.scan_film = True
            params.print = None
            params.settings.neutral_print_filters_from_database = False
            base_paper = "none"
        elif is_bw:
            # Monochrome negative film: print through neutral enlarger without color filtration
            params.io.scan_film = False
            opt_paper = "kodak_2383" if paper_stock == "none" else paper_stock
            params.print = self._load_profile(opt_paper)
            params.settings.neutral_print_filters_from_database = False
            base_paper = opt_paper
        elif print_mode == "scan":
            # Negative film in digital scan mode: digital lab scanner inversion paper (Kodak Ektacolor Edge)
            params.io.scan_film = False
            scan_paper = "kodak_ektacolor_edge"
            params.print = self._load_profile(scan_paper)
            params.settings.neutral_print_filters_from_database = True
            base_paper = scan_paper
        else:
            # Negative film in optical print mode: optical enlarger onto positive photographic paper
            params.io.scan_film = False
            opt_paper = "kodak_2383" if paper_stock == "none" else paper_stock
            params.print = self._load_profile(opt_paper)
            params.settings.neutral_print_filters_from_database = True
            base_paper = opt_paper

        # CRITICAL: Enable lut_mode for deterministic, un-biased 3D LUT generation
        # This completely disables auto_exposure on the synthetic cube lattice,
        # preventing the catastrophic -1.44 EV darkening!
        params.debug.lut_mode = True
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = 0.0

        params.settings.preview_mode = True
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.settings.lut_resolution = 17

        # Enlarger illuminant
        params.enlarger.illuminant = illuminant

        # DIR Couplers: only active for color negative films that actually possess DIR chemistry
        has_dir = True
        if is_bw or is_pos:
            has_dir = False
        else:
            phys_ext = getattr(params.film.data, "physical_extensions", None) or {}
            c_ext = phys_ext.get("couplers") if isinstance(phys_ext, dict) else None
            if isinstance(c_ext, dict):
                cinh = c_ext.get("coupler_inhibition")
                cgeom = c_ext.get("coupler_geometry")
                if cinh is not None and np.sum(np.abs(cinh)) < 1e-6 and not cgeom:
                    has_dir = False

        params.film_render.dir_couplers.active = (dir_amt > 0.01 and has_dir)
        params.film_render.dir_couplers.amount = dir_amt
        params.film_render.dir_couplers.inhibition_interlayer = dir_intl
        params.film_render.dir_couplers.inhibition_samelayer = dir_same

        # Morph Curves
        from spektrafilm.utils.morph_curves import PrintCurvesMorphParams
        params.print_render.density_curves_morph = PrintCurvesMorphParams(
            active=(abs(morph_gamma - 1.0) > 0.01 or dev_exh > 0.01),
            gamma_factor=morph_gamma,
            developer_exhaustion=dev_exh
        )

        if is_bw:
            base_filter = [0.0, 0.0, 0.0]
        else:
            base_filter = self.get_neutral_filter(base_paper, film_stock)
        params.enlarger.c_filter_neutral = float(base_filter[0])
        params.enlarger.m_filter_neutral = float(base_filter[1])
        params.enlarger.y_filter_neutral = float(base_filter[2])
        params.settings.neutral_print_filters_from_database = False
        params.film_render.halation.active = False
        params.film_render.grain.active = False

        digested = digest_params(params)
        sim = Simulator(digested)
        lut_output = sim.process(flat_input)
        lut_3d = np.clip(lut_output.reshape(lut_size, lut_size, lut_size, 3), 0.0, 1.0).astype(np.float32)

        if is_bw:
            # Enforce strictly neutral monochrome output (Rec.709 luminance)
            luma = 0.2126 * lut_3d[..., 0] + 0.7152 * lut_3d[..., 1] + 0.0722 * lut_3d[..., 2]
            lut_3d[..., 0] = luma
            lut_3d[..., 1] = luma
            lut_3d[..., 2] = luma
        elif is_pos:
            lut_3d = self._postprocess_positive_lut(lut_3d, film_stock)

        self._lut_cache[cache_key] = lut_3d
        if is_default_params:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_file, lut_3d)
            except Exception:
                pass
        print(f"[SpektraEngine] 3D LUT 生成完毕，耗时: {round((time.time() - t0)*1000, 1)}ms")
        return lut_3d

    RAW_EXTS = RAW_EXTENSIONS

    @classmethod
    def _extract_exif(cls, file_path):
        meta = {
            "capture_time": "-",
            "resolution": "-",
            "file_size": "-",
            "camera_make": "-",
            "camera_model": "-",
            "serial_number": "-",
            "lens_make": "-",
            "lens_model": "-",
            "iso": "-",
            "aperture": "-",
            "shutter_speed": "-",
            "exposure_bias": "-",
            "exposure_program": "-",
            "metering_mode": "-",
            "flash": "-",
            "focal_length": "-",
            "focal_length_35mm": "-",
            "white_balance": "-",
            "artist": "-",
            "copyright": "-",
            "gps_coords": "-",
            "gps_datum": "-"
        }
        
        # File Size
        try:
            sz_bytes = os.path.getsize(file_path)
            meta["file_size"] = f"{sz_bytes / (1024.0 * 1024.0):.1f} MB"
        except Exception:
            pass

        ext = os.path.splitext(file_path)[1].lower()

        # Fast direct sensor resolution extraction for RAW camera files (<1ms)
        if ext in cls.RAW_EXTS and rawpy is not None:
            try:
                with rawpy.imread(file_path) as raw_probe:
                    rw, rh = raw_probe.sizes.width, raw_probe.sizes.height
                    if getattr(raw_probe.sizes, "flip", 0) in (4, 5, 6, 7):
                        rw, rh = rh, rw
                    if rw > 0 and rh > 0:
                        meta["resolution"] = f"{rw} × {rh}"
            except Exception:
                pass

        # Helper mapping tables
        prog_map = {
            0: "未定义", 1: "手动 (M)", 2: "程序自动 (P)", 3: "光圈优先 (A)",
            4: "快门优先 (S)", 5: "创意程序", 6: "动作程序", 7: "人像模式", 8: "风景模式"
        }
        meter_map = {
            0: "未知", 1: "平均测光", 2: "中央重点平均", 3: "点测光",
            4: "多点测光", 5: "图案 / 多重测光", 6: "局部测光"
        }

        def _format_gps_dms(coord, ref):
            if not coord or len(coord) < 3:
                return ""
            def _val(x):
                if isinstance(x, (tuple, list)) and len(x) == 2 and x[1] != 0:
                    return float(x[0]) / float(x[1])
                return float(x)
            try:
                deg = _val(coord[0])
                mn = _val(coord[1])
                sec = _val(coord[2])
                ref_s = str(ref).strip() if ref else ""
                return f"{int(deg)}°{int(mn)}'{sec:.1f}\" {ref_s}".strip()
            except Exception:
                return ""

        # 1. Fujifilm RAF extraction via embedded JPEG preview header (offset 84:88)
        if ext == ".raf":
            try:
                with open(file_path, "rb") as rf:
                    hdr = rf.read(128)
                    if hdr.startswith(b"FUJIFILMCCD-RAW"):
                        import struct
                        off_jpg = struct.unpack(">I", hdr[84:88])[0]
                        len_jpg = struct.unpack(">I", hdr[88:92])[0]
                        rf.seek(off_jpg)
                        jpg_bytes = rf.read(len_jpg)
                        from PIL import Image, ExifTags
                        with Image.open(io.BytesIO(jpg_bytes)) as j_img:
                            ex = j_img.getexif()
                            if ex:
                                if meta["camera_make"] == "-": meta["camera_make"] = str(ex.get(0x010f, "-")).strip()
                                if meta["camera_model"] == "-": meta["camera_model"] = str(ex.get(0x0110, "-")).strip()
                                if meta["artist"] == "-": meta["artist"] = str(ex.get(0x013b, "-")).strip()
                                if meta["copyright"] == "-": meta["copyright"] = str(ex.get(0x8298, "-")).strip()
                                if meta["capture_time"] == "-": meta["capture_time"] = str(ex.get(0x0132, "-")).strip()
                                try:
                                    sub = ex.get_ifd(0x8769)
                                    if len(sub) > 0:
                                        dt_orig = sub.get(0x9003) or sub.get(0x9004)
                                        if dt_orig:
                                            meta["capture_time"] = str(dt_orig).strip()
                                        if meta["lens_model"] == "-":
                                            raw_lens = sub.get(0xa434, "-")
                                            meta["lens_model"] = str(raw_lens).strip().strip("\x00").strip()
                                        if meta["lens_make"] == "-":
                                            raw_lmake = sub.get(0xa433, "-")
                                            meta["lens_make"] = str(raw_lmake).strip().strip("\x00").strip()
                                        if meta["serial_number"] == "-":
                                            raw_sn = sub.get(0xa431, "-")
                                            meta["serial_number"] = str(raw_sn).strip().strip("\x00").strip()
                                        if meta["focal_length_35mm"] == "-":
                                            fl_35 = sub.get(0xa405)
                                            if fl_35: meta["focal_length_35mm"] = f"{fl_35} 毫米"
                                        if meta["iso"] == "-":
                                            iso_v = sub.get(0x8827)
                                            if iso_v is not None: meta["iso"] = f"ISO-{iso_v}"
                                        if meta["aperture"] == "-":
                                            fn = sub.get(0x829d)
                                            if fn is not None: meta["aperture"] = f"f/{float(fn):.1f}"
                                        if meta["shutter_speed"] == "-":
                                            et = sub.get(0x829a)
                                            if et is not None:
                                                fv = float(et)
                                                meta["shutter_speed"] = f"1/{int(round(1.0/fv))} 秒" if (fv < 1.0 and fv > 0) else f"{fv:.1f} 秒"
                                        if meta["focal_length"] == "-":
                                            fl = sub.get(0x920a)
                                            if fl is not None: meta["focal_length"] = f"{int(round(float(fl)))} 毫米"
                                        if meta["exposure_bias"] == "-":
                                            eb = sub.get(0x9204)
                                            if eb is not None:
                                                bias = float(eb)
                                                meta["exposure_bias"] = f"{bias:+.1f} 档光圈" if bias != 0 else "0 档光圈"
                                        if meta["exposure_program"] == "-":
                                            ep = sub.get(0x8822)
                                            if ep is not None: meta["exposure_program"] = prog_map.get(int(ep), str(ep))
                                        if meta["metering_mode"] == "-":
                                            mm = sub.get(0x9207)
                                            if mm is not None: meta["metering_mode"] = meter_map.get(int(mm), str(mm))
                                        if meta["flash"] == "-":
                                            fl_val = sub.get(0x9209)
                                            if fl_val is not None:
                                                fst = int(fl_val)
                                                meta["flash"] = "开启闪光" if (fst & 1) else ("无闪光，强制" if (fst & 16) else "无闪光")
                                except Exception:
                                    pass
                                try:
                                    gps = ex.get_ifd(0x8825)
                                    if len(gps) > 0:
                                        datum = gps.get(0x0012)
                                        if datum and meta["gps_datum"] == "-": meta["gps_datum"] = str(datum).strip()
                                        lat = gps.get(0x0002)
                                        lat_ref = gps.get(0x0001)
                                        lon = gps.get(0x0004)
                                        lon_ref = gps.get(0x0003)
                                        alt = gps.get(0x0006)
                                        if lat and lon:
                                            c_str = f"{_format_gps_dms(lat, lat_ref)}, {_format_gps_dms(lon, lon_ref)}"
                                            if alt:
                                                alt_val = alt[0]/float(alt[1]) if isinstance(alt, tuple) and alt[1] != 0 else float(alt)
                                                c_str += f" (海拔 {int(round(alt_val))}m)"
                                            meta["gps_coords"] = c_str
                                except Exception:
                                    pass
            except Exception:
                pass

        # 2. Try tifffile (fast for ARW, DNG, CR2, NEF, TIFF)
        try:
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                for page in tif.pages:
                    if meta["resolution"] == "-" and hasattr(page, "shape") and len(page.shape) >= 2:
                        h, w = page.shape[:2]
                        meta["resolution"] = f"{w} × {h}"

                    for tag in page.tags:
                        t_name = tag.name
                        val = tag.value
                        if t_name == 'Make' and meta["camera_make"] == "-":
                            meta["camera_make"] = str(val).strip()
                        elif t_name == 'Model' and meta["camera_model"] == "-":
                            meta["camera_model"] = str(val).strip()
                        elif t_name == 'DateTime' and meta["capture_time"] == "-":
                            meta["capture_time"] = str(val).strip()
                        elif t_name == 'Artist' and meta["artist"] == "-":
                            meta["artist"] = str(val).strip()
                        elif t_name == 'Copyright' and meta["copyright"] == "-":
                            meta["copyright"] = str(val).strip()
                        elif t_name == 'BodySerialNumber' and meta["serial_number"] == "-":
                            meta["serial_number"] = str(val).strip()
                        elif t_name == 'ExifTag' and isinstance(val, dict):
                            sub = val
                            dt_orig = sub.get('DateTimeOriginal') or sub.get(0x9003) or sub.get('DateTimeDigitized') or sub.get(0x9004)
                            if dt_orig:
                                meta["capture_time"] = str(dt_orig).strip()
                            for k, v in sub.items():
                                if k in ('LensModel', 0xa434) and meta["lens_model"] == "-":
                                    meta["lens_model"] = str(v).strip().strip("\x00")
                                elif k in ('LensMake', 0xa433) and meta["lens_make"] == "-":
                                    meta["lens_make"] = str(v).strip().strip("\x00")
                                elif k in ('BodySerialNumber', 0xa431) and meta["serial_number"] == "-":
                                    meta["serial_number"] = str(v).strip().strip("\x00")
                                elif k in ('FocalLengthIn35mmFilm', 0xa405) and meta["focal_length_35mm"] == "-":
                                    meta["focal_length_35mm"] = f"{v} 毫米"
                                elif k in ('FNumber', 0x829d) and meta["aperture"] == "-":
                                    num = v[0] / float(v[1]) if isinstance(v, tuple) and v[1] != 0 else v
                                    meta["aperture"] = f"f/{num:.1f}" if isinstance(num, (int, float)) else str(num)
                                elif k in ('ExposureTime', 0x829a) and meta["shutter_speed"] == "-":
                                    if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                                        meta["shutter_speed"] = f"1/{int(v[1]/v[0])} 秒" if v[0] <= 1 else f"{v[0]/float(v[1]):.2f} 秒"
                                    else:
                                        meta["shutter_speed"] = f"{v} 秒"
                                elif k in ('ISOSpeedRatings', 'PhotographicSensitivity', 0x8827) and meta["iso"] == "-":
                                    meta["iso"] = f"ISO-{v}"
                                elif k in ('FocalLength', 0x920a) and meta["focal_length"] == "-":
                                    num = v[0] / float(v[1]) if isinstance(v, tuple) and v[1] != 0 else v
                                    meta["focal_length"] = f"{int(round(num))} 毫米" if isinstance(num, (int, float)) else str(num)
                                elif k in ('ExposureBiasValue', 0x9204) and meta["exposure_bias"] == "-":
                                    bias = v[0] / float(v[1]) if isinstance(v, tuple) and v[1] != 0 else float(v)
                                    meta["exposure_bias"] = f"{bias:+.1f} 档光圈" if bias != 0 else "0 档光圈"
                                elif k in ('ExposureProgram', 0x8822) and meta["exposure_program"] == "-":
                                    meta["exposure_program"] = prog_map.get(int(v), str(v))
                                elif k in ('MeteringMode', 0x9207) and meta["metering_mode"] == "-":
                                    meta["metering_mode"] = meter_map.get(int(v), str(v))
                                elif k in ('Flash', 0x9209) and meta["flash"] == "-":
                                    fst = int(v)
                                    meta["flash"] = "开启闪光" if (fst & 1) else ("无闪光，强制" if (fst & 16) else "无闪光")
                        elif t_name == 'GPSTag' and isinstance(val, dict):
                            gps = val
                            datum = gps.get('GPSMapDatum') or gps.get(0x0012)
                            if datum and meta["gps_datum"] == "-":
                                meta["gps_datum"] = str(datum).strip()
                            lat = gps.get('GPSLatitude') or gps.get(0x0002)
                            lat_ref = gps.get('GPSLatitudeRef') or gps.get(0x0001)
                            lon = gps.get('GPSLongitude') or gps.get(0x0004)
                            lon_ref = gps.get('GPSLongitudeRef') or gps.get(0x0003)
                            alt = gps.get('GPSAltitude') or gps.get(0x0006)
                            if lat and lon:
                                c_str = f"{_format_gps_dms(lat, lat_ref)}, {_format_gps_dms(lon, lon_ref)}"
                                if alt:
                                    alt_val = alt[0]/float(alt[1]) if isinstance(alt, tuple) and alt[1] != 0 else float(alt)
                                    c_str += f" (海拔 {int(round(alt_val))}m)"
                                meta["gps_coords"] = c_str
        except Exception:
            pass

        # 3. Try PIL Image for standard JPEG / PNG / TIFF fallback
        try:
            from PIL import Image, ExifTags
            with Image.open(file_path) as img:
                if meta["resolution"] == "-":
                    meta["resolution"] = f"{img.width} × {img.height}"
                exif = img.getexif()
                if exif:
                    for k, v in exif.items():
                        tag_name = ExifTags.TAGS.get(k, str(k))
                        if tag_name == "Make" and meta["camera_make"] == "-": meta["camera_make"] = str(v).strip()
                        elif tag_name == "Model" and meta["camera_model"] == "-": meta["camera_model"] = str(v).strip()
                        elif tag_name == "DateTime" and meta["capture_time"] == "-": meta["capture_time"] = str(v).strip()
                        elif tag_name == "Artist" and meta["artist"] == "-": meta["artist"] = str(v).strip()
                        elif tag_name == "Copyright" and meta["copyright"] == "-": meta["copyright"] = str(v).strip()
                    try:
                        sub_exif = exif.get_ifd(0x8769)
                        if len(sub_exif) > 0:
                            dt_orig = sub_exif.get(0x9003) or sub_exif.get(0x9004)
                            if dt_orig:
                                meta["capture_time"] = str(dt_orig).strip()
                            for k, v in sub_exif.items():
                                t = ExifTags.TAGS.get(k, str(k))
                                if t == "LensModel" and meta["lens_model"] == "-": meta["lens_model"] = str(v).strip()
                                elif t == "LensMake" and meta["lens_make"] == "-": meta["lens_make"] = str(v).strip()
                                elif t == "BodySerialNumber" and meta["serial_number"] == "-": meta["serial_number"] = str(v).strip()
                                elif t == "FocalLengthIn35mmFilm" and meta["focal_length_35mm"] == "-": meta["focal_length_35mm"] = f"{v} 毫米"
                                elif t == "FNumber" and meta["aperture"] == "-": meta["aperture"] = f"f/{float(v):.1f}"
                                elif t == "ExposureTime" and meta["shutter_speed"] == "-":
                                    fv = float(v)
                                    meta["shutter_speed"] = f"1/{int(round(1.0/fv))} 秒" if fv < 1.0 and fv > 0 else f"{fv:.1f} 秒"
                                elif t in ("ISOSpeedRatings", "PhotographicSensitivity") and meta["iso"] == "-": meta["iso"] = f"ISO-{v}"
                                elif t == "FocalLength" and meta["focal_length"] == "-": meta["focal_length"] = f"{int(round(float(v)))} 毫米"
                                elif t == "ExposureBiasValue" and meta["exposure_bias"] == "-":
                                    b = float(v)
                                    meta["exposure_bias"] = f"{b:+.1f} 档光圈" if b != 0 else "0 档光圈"
                                elif t == "ExposureProgram" and meta["exposure_program"] == "-":
                                    meta["exposure_program"] = prog_map.get(int(v), str(v))
                                elif t == "MeteringMode" and meta["metering_mode"] == "-":
                                    meta["metering_mode"] = meter_map.get(int(v), str(v))
                                elif t == "Flash" and meta["flash"] == "-":
                                    fst = int(v)
                                    meta["flash"] = "开启闪光" if (fst & 1) else ("无闪光，强制" if (fst & 16) else "无闪光")
                    except Exception:
                        pass
                    try:
                        gps = exif.get_ifd(0x8825)
                        if len(gps) > 0:
                            datum = gps.get(0x0012)
                            if datum and meta["gps_datum"] == "-": meta["gps_datum"] = str(datum).strip()
                            lat = gps.get(0x0002)
                            lat_ref = gps.get(0x0001)
                            lon = gps.get(0x0004)
                            lon_ref = gps.get(0x0003)
                            alt = gps.get(0x0006)
                            if lat and lon:
                                c_str = f"{_format_gps_dms(lat, lat_ref)}, {_format_gps_dms(lon, lon_ref)}"
                                if alt:
                                    alt_val = alt[0]/float(alt[1]) if isinstance(alt, tuple) and alt[1] != 0 else float(alt)
                                    c_str += f" (海拔 {int(round(alt_val))}m)"
                                meta["gps_coords"] = c_str
                    except Exception:
                        pass
        except Exception:
            pass

        # 4. Rawpy extraction for white balance, color temp, camera make, model, resolution
        if rawpy is not None and ext in cls.RAW_EXTS:
            try:
                with open(file_path, "rb") as fp:
                    with rawpy.imread(fp) as raw:
                        if meta["resolution"] == "-":
                            rw, rh = raw.sizes.width, raw.sizes.height
                            if getattr(raw.sizes, "flip", 0) in (4, 5, 6, 7):
                                rw, rh = rh, rw
                            meta["resolution"] = f"{rw} × {rh}"

                        wb = getattr(raw, "camera_whitebalance", None)
                        mat = getattr(raw, "rgb_xyz_matrix", None)
                        if wb is not None and len(wb) >= 3 and wb[0] > 0 and wb[1] > 0 and wb[2] > 0:
                            calc_cct = None
                            calc_duv = 0.0

                            # Method A: Invert camera RGB->XYZ matrix and evaluate neutral sensor chromaticity
                            if mat is not None and hasattr(mat, "shape") and mat.shape[0] >= 3 and mat.shape[1] >= 3:
                                m3 = mat[:3, :3]
                                if np.linalg.matrix_rank(m3) == 3:
                                    try:
                                        import colour
                                        inv_m = np.linalg.inv(m3)
                                        cam_n = np.array([1.0 / wb[0], 1.0 / wb[1], 1.0 / wb[2]])
                                        xyz = np.dot(inv_m, cam_n)
                                        if xyz[1] != 0:
                                            xyz = xyz / xyz[1]
                                        cct_v, duv_v = colour.temperature.XYZ_to_CCT_Ohno2013(xyz)
                                        calc_cct = float(cct_v)
                                        calc_duv = float(duv_v)
                                    except Exception:
                                        pass

                            # Method B: Daylight reference white balance ratio fallback
                            if calc_cct is None:
                                day_wb = getattr(raw, "daylight_whitebalance", None)
                                if day_wb is not None and len(day_wb) >= 3 and day_wb[0] > 0 and day_wb[2] > 0:
                                    ratio_rb = (wb[0] / wb[2]) / (day_wb[0] / day_wb[2])
                                    calc_cct = 5500.0 / np.sqrt(max(0.1, ratio_rb))

                            if calc_cct is not None:
                                cct_rounded = round(float(calc_cct) / 25.0) * 25.0
                                cct_clamped = max(3000.0, min(8500.0, cct_rounded))
                                tint_val = round(float(1.0 + calc_duv * 4.0), 2)
                                tint_clamped = max(0.8, min(1.2, tint_val))
                                meta["color_temp"] = cct_clamped
                                meta["tint"] = tint_clamped
                                meta["white_balance"] = f"{int(cct_clamped)} K"
            except Exception:
                pass

        if "color_temp" not in meta:
            meta["color_temp"] = 5500.0
            meta["tint"] = 1.0
            meta["white_balance"] = "5500 K"

        return meta

    @staticmethod
    def calculate_auto_exposure_ev(float_img):
        """Calculate recommended base exposure compensation using Fotufilm-calibrated 18% middle-gray & highlight meter."""
        if float_img is None or not isinstance(float_img, np.ndarray):
            return 0.0
        try:
            from spektrafilm.utils.autoexposure import measure_autoexposure_ev
            h, w = float_img.shape[:2]
            max_dim = 256
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
                sample_img = cv2.resize(float_img, (sw, sh), interpolation=cv2.INTER_AREA)
            else:
                sample_img = float_img
            if float(np.nanmean(sample_img)) < 1e-4:
                return 0.0

            # 1. Standard center-weighted EV
            ev_center = measure_autoexposure_ev(sample_img, color_space='ProPhoto RGB', apply_cctf_decoding=False, method='center_weighted')
            if np.isnan(ev_center) or np.isinf(ev_center):
                ev_center = 0.0

            # 2. Adaptive highlight & median anchor for high dynamic range scenes (e.g. mobile RAW with bright sky)
            # Reference: Fotufilm retimeHighlight & toneKey algorithm
            Y = 0.288040 * sample_img[..., 0] + 0.711874 * sample_img[..., 1] + 0.000086 * sample_img[..., 2]
            valid = Y[Y > 1e-5]
            if len(valid) > 100:
                stops = np.log2(valid / 0.184)
                p_med = float(np.percentile(stops, 50))
                p_bright = float(np.percentile(stops, 99.5))
                span = p_bright - p_med

                # Fotufilm retimeHighlight: diffuse highlight placed 2.5..3.5 stops over median
                white = min(max(p_bright, p_med + 2.5), p_med + 3.5)
                ev_hl = 3.0 - white
                ev_med = -0.67 - p_med
                ev_target = min(ev_med, ev_hl)

                if span > 2.5 and ev_target > ev_center:
                    t = float(np.clip((span - 2.5) / 1.5, 0.0, 1.0))
                    blend = t * t * (3.0 - 2.0 * t)
                    ev_final = (1.0 - blend) * ev_center + blend * ev_target
                else:
                    ev_final = ev_center
            else:
                ev_final = ev_center

            return float(np.clip(round(ev_final, 2), -3.0, 3.0))
        except Exception as e:
            print(f"[SpektraEngine] 测光计算异常: {e}")
            return 0.0

    @classmethod
    def decode_photo_data(cls, file_path, preview_max_edge=2048, is_first=True, load_preview=None, fast_thumbnail=False):
        """Thread-safe, stateless worker function to decode RAW or SDR image,
        compute base_ev, extract EXIF, and generate thumbnail and preview arrays.
        Safe for concurrent execution across high-CPU worker pools.
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"文件不存在: {file_path}"}

            ext = os.path.splitext(file_path)[1].lower()
            original_meta = {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "ext": ext
            }
            # Extract rich EXIF shooting attributes
            original_meta.update(cls._extract_exif(file_path))

            raw_exts = cls.RAW_EXTS

            if ext in raw_exts:
                if rawpy is None:
                    return {"success": False, "error": "系统未检测到 rawpy 模块，无法解析相机 RAW"}

                if fast_thumbnail and (not is_first) and (load_preview is not True):
                    # Fast path: Extract camera-embedded JPEG preview in sub-15ms
                    # Avoids expensive LibRaw full-resolution demosaicing during batch import
                    try:
                        with open(file_path, "rb") as fp:
                            with rawpy.imread(fp) as raw:
                                rw, rh = raw.sizes.width, raw.sizes.height
                                flip = getattr(raw.sizes, "flip", 0)
                                if flip in (4, 5, 6, 7):
                                    w, h = rh, rw
                                else:
                                    w, h = rw, rh
                                original_meta["width"] = w
                                original_meta["height"] = h
                                original_meta["resolution"] = f"{w} × {h}"

                                thumb = raw.extract_thumb()
                                if thumb.format == rawpy.ThumbFormat.JPEG:
                                    file_bytes = np.frombuffer(thumb.data, dtype=np.uint8)
                                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                                    if img_bgr is not None:
                                        th_h_raw, th_w_raw = img_bgr.shape[:2]
                                        if (h > w and th_w_raw > th_h_raw):
                                            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                                        elif (w > h and th_h_raw > th_w_raw):
                                            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

                                        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                        th_h = 72
                                        th_w = max(48, int(rgb.shape[1] * (th_h / float(rgb.shape[0]))))
                                        th_u8 = cv2.resize(rgb, (th_w, th_h), interpolation=cv2.INTER_AREA)

                                        from . import session_cache_manager
                                        cache_entry = {
                                            "path": file_path,
                                            "filename": original_meta["filename"],
                                            "width": w,
                                            "height": h,
                                            "thumbnail_rgb": th_u8,
                                            "raw_thumbnail_rgb": th_u8,
                                            "graded_thumbnail_rgb": None,
                                            "exif": original_meta,
                                            "auto_ev": 0.0,
                                            "base_ev": 0.0,
                                            "is_raw": True
                                        }
                                        session_cache_manager.save_photo_cache(file_path, cache_entry, save_preview=False)

                                        return {
                                            "success": True,
                                            "path": file_path,
                                            "filename": original_meta["filename"],
                                            "width": w,
                                            "height": h,
                                            "preview_w": th_w,
                                            "preview_h": th_h,
                                            "thumbnail_rgb": th_u8,
                                            "raw_thumbnail_rgb": th_u8,
                                            "graded_thumbnail_rgb": None,
                                            "float_img": None,
                                            "raw_preview": None,
                                            "exif": dict(original_meta),
                                            "is_raw": True,
                                            "auto_ev": 0.0,
                                            "base_ev": 0.0
                                        }
                    except Exception as e_fast:
                        logger.debug(f"Fast thumbnail extraction fell back to demosaic: {e_fast}")

                # Use fast half_size demosaic for background batch loading to prevent OOM across worker pools
                use_half = (not is_first) and (load_preview is not True) and (preview_max_edge > 0 and preview_max_edge <= 2048)
                demosaic_algo = rawpy.DemosaicAlgorithm.LINEAR if use_half else rawpy.DemosaicAlgorithm.AHD

                with open(file_path, "rb") as fp:
                    with rawpy.imread(fp) as raw:
                        rw, rh = raw.sizes.width, raw.sizes.height
                        flip = getattr(raw.sizes, "flip", 0)
                        if flip in (4, 5, 6, 7):
                            w, h = rh, rw
                        else:
                            w, h = rw, rh
                        original_meta["width"] = w
                        original_meta["height"] = h
                        original_meta["resolution"] = f"{w} × {h}"
                        try:
                            # 16-bit linear ProPhoto extraction using LibRaw native ProPhoto colorspace
                            rgb16 = raw.postprocess(
                                demosaic_algorithm=demosaic_algo,
                                half_size=use_half,
                                four_color_rgb=False,
                                output_color=rawpy.ColorSpace.ProPhoto,
                                output_bps=16,
                                gamma=(1, 1),
                                no_auto_bright=True,
                                no_auto_scale=False,
                                use_camera_wb=True,
                                highlight_mode=rawpy.HighlightMode.Clip
                            )
                        except Exception:
                            rgb16 = raw.postprocess(
                                demosaic_algorithm=demosaic_algo,
                                half_size=use_half,
                                four_color_rgb=False,
                                output_color=rawpy.ColorSpace.ProPhoto,
                                output_bps=16,
                                gamma=(1, 1),
                                no_auto_bright=True,
                                no_auto_scale=False,
                                use_auto_wb=True,
                                highlight_mode=rawpy.HighlightMode.Clip
                            )
                # Linear ProPhoto float32 representation
                rgb_linear_pro = (rgb16.astype(np.float32) / 65535.0)
                del rgb16
                # Ensure original_meta orientation matches demosaiced array
                # (without shrinking full resolution to half-size when use_half is enabled)
                h_act, w_act = rgb_linear_pro.shape[:2]
                if (h_act > w_act and w > h) or (w_act > h_act and h > w):
                    w, h = h, w
                    original_meta["width"] = w
                    original_meta["height"] = h
                    original_meta["resolution"] = f"{w} × {h}"
            else:
                img_bgr = None
                try:
                    with open(file_path, "rb") as fp:
                        file_bytes = np.frombuffer(fp.read(), dtype=np.uint8)
                        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                except Exception:
                    img_bgr = None

                if img_bgr is None:
                    try:
                        from PIL import Image
                        with Image.open(file_path) as pil_im:
                            pil_arr = np.array(pil_im)
                            if len(pil_arr.shape) == 2:
                                pil_arr = cv2.cvtColor(pil_arr, cv2.COLOR_GRAY2RGB)
                            elif pil_arr.shape[2] == 4:
                                pil_arr = cv2.cvtColor(pil_arr, cv2.COLOR_RGBA2RGB)
                            rgb_raw = pil_arr
                    except Exception:
                        try:
                            import tifffile
                            tif_arr = tifffile.imread(file_path)
                            if len(tif_arr.shape) == 2:
                                tif_arr = cv2.cvtColor(tif_arr, cv2.COLOR_GRAY2RGB)
                            elif tif_arr.shape[2] == 4:
                                tif_arr = cv2.cvtColor(tif_arr, cv2.COLOR_RGBA2RGB)
                            rgb_raw = tif_arr
                        except Exception as e_tif:
                            return {"success": False, "error": f"无法解码该图片文件: {e_tif}"}
                else:
                    if len(img_bgr.shape) == 2:
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
                    elif img_bgr.shape[2] == 4:
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
                    rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                h, w = rgb_raw.shape[:2]
                if np.issubdtype(rgb_raw.dtype, np.floating):
                    f_img = np.clip(rgb_raw.astype(np.float32), 0.0, 1.0)
                elif rgb_raw.dtype == np.uint16:
                    f_img = rgb_raw.astype(np.float32) / 65535.0
                else:
                    f_img = rgb_raw.astype(np.float32) / 255.0
                del rgb_raw
                # Convert standard sRGB image to linear ProPhoto RGB
                lin_srgb = srgb_to_linear(f_img)
                rgb_linear_pro = np.maximum(0.0, lin_srgb @ SRGB_TO_PROPHOTO.T)
                del f_img, lin_srgb

            original_meta["width"] = w
            original_meta["height"] = h
            original_meta["resolution"] = f"{w} × {h}"

            # Downscale smoothly after full-fidelity demosaic for GPU preview texture (1080P/2K/4K/Full)
            cur_h, cur_w = rgb_linear_pro.shape[:2]
            max_edge = 0 if preview_max_edge == 0 else (preview_max_edge if (preview_max_edge is not None and preview_max_edge > 0) else 2048)
            if max_edge > 0 and max(cur_h, cur_w) > max_edge:
                scale = max_edge / float(max(cur_h, cur_w))
                new_w = int(cur_w * scale)
                new_h = int(cur_h * scale)
                preview_small = cv2.resize(rgb_linear_pro, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                preview_small = rgb_linear_pro
            del rgb_linear_pro

            is_raw_file = (ext in raw_exts)
            if is_raw_file:
                # Calculate auto-exposure baseline for metadata recommendation
                auto_ev = cls.calculate_auto_exposure_ev(preview_small)
                base_ev = auto_ev
                # Retain pristine scene-linear ProPhoto radiance values without pre-multiplying base_ev
                raw_preview = preview_small.astype(np.float32)
            else:
                auto_ev = 0.0
                base_ev = 0.0
                raw_preview = preview_small.astype(np.float32)
            del preview_small

            # Small filmstrip negative thumbnail (height 72px) with smooth highlight roll-off
            th_h = 72
            th_w = max(48, int(raw_preview.shape[1] * (th_h / float(raw_preview.shape[0]))))
            th_img = cv2.resize(raw_preview, (th_w, th_h), interpolation=cv2.INTER_AREA)
            th_u8 = (prophoto_to_srgb_filmic(th_img) * 255.0).astype(np.uint8)

            # Save through to session cache for sub-100ms subsequent loads
            from . import session_cache_manager
            cache_entry = {
                "path": file_path,
                "filename": original_meta["filename"],
                "width": w,
                "height": h,
                "thumbnail_rgb": th_u8,
                "raw_thumbnail_rgb": th_u8,
                "graded_thumbnail_rgb": None,
                "exif": original_meta,
                "auto_ev": auto_ev,
                "base_ev": base_ev,
                "is_raw": is_raw_file
            }
            session_cache_manager.save_photo_cache(file_path, cache_entry, preview_array=raw_preview, save_preview=True)

            should_keep = (load_preview is True) if load_preview is not None else bool(is_first)
            out_dict = {
                "success": True,
                "path": file_path,
                "filename": original_meta["filename"],
                "width": w,
                "height": h,
                "preview_w": raw_preview.shape[1],
                "preview_h": raw_preview.shape[0],
                "thumbnail_rgb": th_u8,
                "raw_thumbnail_rgb": th_u8,
                "graded_thumbnail_rgb": None,
                "float_img": raw_preview if should_keep else None,
                "raw_preview": raw_preview if should_keep else None,
                "exif": dict(original_meta),
                "is_raw": is_raw_file,
                "auto_ev": auto_ev,
                "base_ev": base_ev
            }
            if not should_keep:
                del raw_preview
            return out_dict
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"解析图片或 RAW 发生错误: {str(e)}"}

    def load_image(self, file_path):
        """Load RAW or high-fidelity image into float32 array for GPU texture."""
        max_edge = 2048
        try:
            from . import config_manager
            max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
        except Exception:
            max_edge = 2048

        res = self.decode_photo_data(file_path, preview_max_edge=max_edge, is_first=True)
        if not res or not res.get("success"):
            return res or {"success": False, "error": f"无法载入文件: {file_path}"}

        self.current_file_path = file_path
        self.original_meta = dict(res.get("exif") or {})
        self.raw_preview = res.get("raw_preview")

        existing = next((p for p in self.session_photos if p["path"] == file_path), None)
        photo_id = existing["id"] if existing else f"film_{uuid.uuid4().hex[:8]}"

        photo_item = {
            "id": photo_id,
            "path": file_path,
            "filename": res["filename"],
            "width": res["width"],
            "height": res["height"],
            "thumbnail_rgb": res["thumbnail_rgb"],
            "raw_preview": None,
            "meta": dict(self.original_meta),
            "is_raw": res["is_raw"],
            "base_ev": res["base_ev"],
            "auto_ev": res.get("auto_ev", 0.0)
        }

        if existing:
            self.session_photos[self.session_photos.index(existing)] = photo_item
        else:
            self.session_photos.append(photo_item)

        self.active_photo_id = photo_id
        res["id"] = photo_id
        res["photos"] = self.get_session_photos()
        res["active_id"] = photo_id
        print(f"[SpektraEngine] 成功载入 {self.original_meta['filename']}: 原始尺寸 {res['width']}x{res['height']}，GPU 纹理尺寸 {self.raw_preview.shape[1]}x{self.raw_preview.shape[0]}，基准测光: {res['base_ev']:+.2f} EV")
        return res

    def load_multiple_images(self, file_paths):
        results = []
        errors = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            res = self.load_image(path)
            if res.get("success"):
                results.append(res)
            else:
                errors.append(f"{os.path.basename(path)}: {res.get('error')}")

        return {
            "success": len(results) > 0,
            "count": len(results),
            "id": self.active_photo_id,
            "photos": self.get_session_photos(),
            "active_id": self.active_photo_id,
            "errors": errors
        }

    def get_session_photos(self):
        return [
            {
                "id": p["id"],
                "filename": p["filename"],
                "path": p["path"],
                "width": p["width"],
                "height": p["height"],
                "thumbnail_rgb": p["thumbnail_rgb"],
                "is_raw": p["is_raw"],
                "is_active": (p["id"] == self.active_photo_id)
            }
            for p in self.session_photos
        ]

    def set_active_photo(self, photo_id):
        photo = next((p for p in self.session_photos if p["id"] == photo_id), None)
        if not photo:
            return {"success": False, "error": "未找到指定底片"}

        self.active_photo_id = photo_id
        if photo.get("raw_preview") is not None:
            self.raw_preview = photo["raw_preview"]
        self.raw_full = None
        self.original_meta = dict(photo.get("meta") or {})
        self.current_file_path = photo.get("path")

        return {
            "success": True,
            "id": photo["id"],
            "filename": photo["filename"],
            "width": photo["width"],
            "height": photo["height"],
            "photos": self.get_session_photos(),
            "active_id": photo_id
        }

    def remove_photo(self, photo_id):
        idx = next((i for i, p in enumerate(self.session_photos) if p["id"] == photo_id), None)
        if idx is None:
            return {"success": False, "error": "未找到底片"}

        self.session_photos.pop(idx)
        if len(self.session_photos) == 0:
            self.active_photo_id = None
            self.raw_full = None
            self.raw_preview = None
            self.original_meta = {}
            return {"success": True, "photos": [], "active_id": None}

        if self.active_photo_id == photo_id:
            new_idx = min(idx, len(self.session_photos) - 1)
            new_active = self.session_photos[new_idx]
            return self.set_active_photo(new_active["id"])

        return {"success": True, "photos": self.get_session_photos(), "active_id": self.active_photo_id}

    def clear_all_photos(self):
        """Completely flush all session photos and reset active state."""
        self.session_photos.clear()
        self.active_photo_id = None
        self.raw_full = None
        self.raw_preview = None
        self.original_meta = {}
        self.current_file_path = None
        return {"success": True, "photos": [], "active_id": None}

    def _load_full_resolution(self, file_path, base_ev=0.0):
        """Full 16-bit float32 loader for master export on demand in Linear ProPhoto RGB."""
        ext = os.path.splitext(file_path)[1].lower()
        raw_exts = self.RAW_EXTS
        if ext in raw_exts:
            with open(file_path, "rb") as fp:
                with rawpy.imread(fp) as raw:
                    try:
                        rgb16 = raw.postprocess(
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                            half_size=False,
                            four_color_rgb=False,
                            output_color=rawpy.ColorSpace.ProPhoto,
                            output_bps=16,
                            gamma=(1, 1),
                            no_auto_bright=True,
                            no_auto_scale=False,
                            use_camera_wb=True,
                            highlight_mode=rawpy.HighlightMode.Clip
                        )
                    except Exception:
                        rgb16 = raw.postprocess(
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                            half_size=False,
                            four_color_rgb=False,
                            output_color=rawpy.ColorSpace.ProPhoto,
                            output_bps=16,
                            gamma=(1, 1),
                            no_auto_bright=True,
                            no_auto_scale=False,
                            use_auto_wb=True,
                            highlight_mode=rawpy.HighlightMode.Clip
                        )
                    rgb_pro = (rgb16.astype(np.float32) / 65535.0)
                    return rgb_pro
        else:
            img_bgr = None
            try:
                with open(file_path, "rb") as fp:
                    file_bytes = np.frombuffer(fp.read(), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            except Exception:
                img_bgr = None

            if img_bgr is None:
                try:
                    from PIL import Image
                    with Image.open(file_path) as pil_im:
                        img_rgb = np.array(pil_im)
                        if len(img_rgb.shape) == 2:
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                        elif img_rgb.shape[2] == 4:
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
                except Exception:
                    try:
                        import tifffile
                        img_rgb = tifffile.imread(file_path)
                        if len(img_rgb.shape) == 2:
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                        elif img_rgb.shape[2] == 4:
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
                    except Exception as e_tif:
                        raise ValueError(f"无法解码源图像: {e_tif}")
            else:
                if len(img_bgr.shape) == 2:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
                elif img_bgr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if np.issubdtype(img_rgb.dtype, np.floating):
                out = img_rgb.astype(np.float32)
                max_v = float(np.nanmax(out)) if out.size > 0 else 1.0
                if max_v > 255.0:
                    f_img = out / 65535.0
                elif max_v > 1.0:
                    f_img = out / 255.0
                else:
                    f_img = out
            elif img_rgb.dtype == np.uint16:
                f_img = img_rgb.astype(np.float32) / 65535.0
            else:
                f_img = img_rgb.astype(np.float32) / 255.0

            lin_srgb = srgb_to_linear(f_img)
            return np.maximum(0.0, lin_srgb @ SRGB_TO_PROPHOTO.T)

    def export_image(self, params_dict, output_path, format_type="tiff16", quality=9, source_path=None, progress_cb=None, cancel_cb=None):
        """Render at 100% full resolution and write to disk."""
        try:
            if cancel_cb and cancel_cb():
                return {"success": False, "cancelled": True, "error": "用户已取消导出"}

            if progress_cb:
                progress_cb(5, "正在载入底片并准备物理色彩管线...")

            target_src = source_path or params_dict.get("source_path") or self.current_file_path
            raw_target = None
            base_ev = float(params_dict.get("base_ev", 0.0))
            if not base_ev and target_src:
                matched_p = next((p for p in self.session_photos if p["path"] == target_src), None)
                if matched_p and "base_ev" in matched_p:
                    base_ev = float(matched_p["base_ev"])

            if target_src:
                try:
                    print(f"[SpektraEngine] 正在为高精度成片载入 100% 全分辨率底片: {target_src} (基准EV: {base_ev:+.2f})")
                    raw_target = self._load_full_resolution(target_src, base_ev=base_ev)
                except Exception as ex_load:
                    print(f"[SpektraEngine] 全分辨率载入异常，回退至GPU底片: {ex_load}")

            if raw_target is None:
                raw_target = self.raw_full if self.raw_full is not None else self.raw_preview

            if raw_target is None:
                return {"success": False, "error": "没有可导出的图像底片数据"}

            film_stock = params_dict.get("film_stock", "kodak_portra_400")
            paper_stock = params_dict.get("paper_stock", "kodak_2383")
            
            lut_3d = self.get_3d_lut(film_stock, paper_stock, lut_size=33, params_dict=params_dict)

            exp_val = float(params_dict.get("exposure_ev", 0.0))
            temp_val = float(params_dict.get("color_temp", 5500.0))
            tint_val = float(params_dict.get("tint", 1.0))
            is_pos = self.is_positive_film(film_stock)
            if is_pos:
                c_val = 0.0
                m_val = 0.0
                y_val = 0.0
                print_exp = 1.0
                preflash = 0.0
            else:
                c_val = float(params_dict.get("enlarger_cyan", 0.0))
                m_val = float(params_dict.get("enlarger_magenta", 0.0))
                y_val = float(params_dict.get("enlarger_yellow", 0.0))
                print_exp = float(params_dict.get("print_exposure", 1.0))
                preflash = float(params_dict.get("pre_flash", 0.0))
            halation = float(params_dict.get("halation_amount", params_dict.get("halation", 1.0)))
            hal_spatial_scale = float(params_dict.get("halation_spatial_scale", 1.0))
            scatter_amount = float(params_dict.get("scatter_amount", 1.0))
            hal_boost = float(params_dict.get("boost_ev", params_dict.get("halation_boost", 0.0)))
            grain = float(params_dict.get("grain", 0.4))
            grain_size = float(params_dict.get("grain_size", 1.0))
            fmt_mm = float(params_dict.get("film_format_mm", 35.0))
            grain_scale = (35.0 / max(5.0, fmt_mm)) * max(0.2, grain_size)
            diff_family = str(params_dict.get("diffusion_family", "none")).lower()
            diff_strength = float(params_dict.get("diffusion_strength", 0.0))
            diff_warmth = float(params_dict.get("diffusion_warmth", 0.0))

            H, W = raw_target.shape[:2]
            lut_size = lut_3d.shape[0]
            rendered_float = np.empty((H, W, 3), dtype=np.float32)

            if film_stock == "none":
                # Pure digital RAW bypass export: output untouched original raw converted directly to sRGB
                if progress_cb:
                    progress_cb(60, "正在导出数码原片直出 (Bypass)...")
                rendered_float = np.clip(prophoto_to_srgb(raw_target), 0.0, 1.0)
                if progress_cb:
                    progress_cb(92, "正在编码并写入图像文件...")

                if os.path.isdir(output_path) or output_path.endswith(('/', '\\\\')):
                    src = source_path or params_dict.get("source_path") or self.current_file_path or "developed"
                    base_name = os.path.splitext(os.path.basename(src))[0]
                    fmt_str = str(format_type).lower()
                    ext = ".tif" if "tif" in fmt_str else (".png" if "png" in fmt_str else ".jpg")
                    output_path = os.path.join(output_path, f"{base_name}_developed{ext}")

                enc_params = dict(params_dict)
                enc_params["format"] = format_type
                enc_params["quality"] = quality
                enc_params["source_path"] = target_src
                save_encoded_image(rendered_float, output_path, enc_params)

                if progress_cb:
                    progress_cb(100, f"成片扫描完成: {os.path.basename(output_path)}")
                return {"success": True, "output_path": output_path}

            base_temp = float(params_dict.get("base_temp", 5500.0))
            base_tint = float(params_dict.get("base_tint", 1.0))
            temp_factor = (temp_val - base_temp) / 3000.0
            tint_factor = tint_val / max(0.001, base_tint)
            cmy_arr = np.array([c_val, m_val, y_val], dtype=np.float32)
            cmy_factor = 1.0 - cmy_arr * 0.015

            if progress_cb:
                progress_cb(10, "正在计算全局线性曝光与基准白平衡...")

            # 1. Exposure compensation & White balance in linear radiance space
            base_ev = float(params_dict.get("base_ev", 0.0))
            eff_exp = exp_val + base_ev
            linear_full = (raw_target * float(np.exp2(eff_exp))).astype(np.float32)
            linear_full[..., 0] *= (1.0 + temp_factor * 0.22)
            linear_full[..., 2] *= (1.0 - temp_factor * 0.22)
            linear_full[..., 1] *= tint_factor

            # 1.5. Pre-LUT Scene-Linear Highlights & Shadows Tone Shaping (Zone System EV Hermite Spline)
            hl_val = float(params_dict.get("highlights", 0.0))
            sh_val = float(params_dict.get("shadows", 0.0))
            linear_full = apply_highlights_shadows_np(linear_full, hl_val, sh_val)

            if cancel_cb and cancel_cb():
                return {"success": False, "cancelled": True, "error": "用户已取消导出"}

            pixel_size_um = float((fmt_mm * 1000.0) / max(H, W))

            # 2. Optical Diffusion Filter (Aligned with Viewport Vogel Spiral/Gaussian bloom)
            if diff_family != "none" and diff_strength > 0.001:
                if progress_cb:
                    progress_cb(18, f"正在进行光学柔光印相模拟 ({diff_family})...")
                try:
                    linear_full = apply_diffusion_np(linear_full, diff_family, diff_strength, diff_warmth)
                except Exception as e:
                    print(f"[SpektraExport] 柔光模拟异常: {e}")

            # 2.5. Official Spektrafilm Physical Halation & In-Emulsion Scatter in Scene-linear space
            if halation > 0.001 or scatter_amount > 0.001:
                try:
                    if progress_cb:
                        progress_cb(25, "正在模拟真实物理胶片光晕与防光晕层漫射 (Official Spektrafilm Dual-Pass)...")

                    hal_preset = self.get_film_halation_preset(film_stock)
                    hal_strength_rgb = np.array(hal_preset["strength"], dtype=np.float32)
                    hal_sigma_um = float(hal_preset["sigma_h"])
                    eff_strength = hal_strength_rgb * halation

                    # 1. In-emulsion scatter pass: physical micro-diffusion (energy-preserving)
                    if scatter_amount > 0.001:
                        sigma_c_px = max(0.5, (2.0 * max(0.2, hal_spatial_scale)) / pixel_size_um)
                        sigma_t_px = max(1.0, (9.5 * max(0.2, hal_spatial_scale)) / pixel_size_um)
                        kc = int(sigma_c_px * 3.0) * 2 + 1
                        kt = int(sigma_t_px * 3.0) * 2 + 1
                        core = cv2.GaussianBlur(linear_full, (min(kc, 101), min(kc, 101)), sigma_c_px)
                        tail = cv2.GaussianBlur(linear_full, (min(kt, 151), min(kt, 151)), sigma_t_px)
                        w_tail = np.array([0.78, 0.65, 0.67], dtype=np.float32)
                        scattered = core * (1.0 - w_tail) + tail * w_tail
                        eff_s = min(0.5, scatter_amount * 0.06)
                        linear_full = (1.0 - eff_s) * linear_full + eff_s * scattered

                    # 2. Base back-reflection halation pass: Spektrafilm multi-bounce sum with DC normalization
                    if halation > 0.001 and np.any(eff_strength > 0):
                        sigma_h_px = (hal_sigma_um * max(0.2, hal_spatial_scale)) / pixel_size_um
                        decay = np.array([0.5 ** (k - 1) for k in range(1, 4)], dtype=np.float32)
                        decay /= decay.sum()

                        # Highlight boost if enabled (0.0 by default)
                        hal_input = linear_full
                        if hal_boost > 0.001:
                            luma_in = 0.2126 * linear_full[..., 0] + 0.7152 * linear_full[..., 1] + 0.0722 * linear_full[..., 2]
                            boost_scale = 1.0 + np.maximum(0.0, luma_in - 0.7) * (hal_boost * 0.6)
                            hal_input = linear_full * boost_scale[..., None]

                        hal_blur = np.zeros_like(linear_full)
                        for k, wk in zip(range(1, 4), decay):
                            sigma_k = sigma_h_px * np.sqrt(k)
                            ksize = int(sigma_k * 3.0) * 2 + 1
                            ksize = min(ksize, 301)
                            blur_k = cv2.GaussianBlur(hal_input, (ksize, ksize), sigma_k)
                            hal_blur += blur_k * wk

                        # Official Spektrafilm physical halation DC normalization:
                        linear_full = (linear_full + eff_strength * hal_blur) / (1.0 + eff_strength)
                except Exception as e_hal:
                    print(f"[SpektraExport] 光晕模拟异常: {e_hal}")

            if cancel_cb and cancel_cb():
                return {"success": False, "cancelled": True, "error": "用户已取消导出"}

            # Multi-threaded High-Precision CPU Pipeline (3D LUT + DIR Coupler + CMY + Paper Morph + Halation + Pre-LUT Grain)
            import concurrent.futures
            import time
            max_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
            chunk_size = 256
            chunks = [(y0, min(y0 + chunk_size, H)) for y0 in range(0, H, chunk_size)]

            dir_intl = float(params_dict.get("dir_interlayer", 1.0))
            dir_same = float(params_dict.get("dir_samelayer", 1.0))
            morph_gamma = float(params_dict.get("morph_gamma", 1.0))
            dev_exh = float(params_dict.get("developer_exhaustion", 0.0))

            grain_preset = self.get_film_grain_preset(film_stock)
            st_size = float(grain_preset.get("size_mm", 0.005))
            st_strength = float(grain_preset.get("strength", 0.010))
            st_lw = np.array(grain_preset.get("layer_weights", [0.70, 1.0, 1.35]), dtype=np.float32)

            def _process_chunk(y_range):
                y0, y1 = y_range
                chunk_linear = linear_full[y0:y1].copy()

                # Pre-LUT physical transmittance grain (matches GPU shader)
                if grain > 0.01:
                    eff_strength = (grain / 0.40) * (st_strength / 0.010)
                    lin_luma = 0.2126 * chunk_linear[..., 0] + 0.7152 * chunk_linear[..., 1] + 0.0722 * chunk_linear[..., 2]
                    dens_mod = (1.0 - np.exp(-lin_luma * 15.0)) / (1.0 + (lin_luma / 0.28) ** 2.2)
                    noise = (np.random.randn(chunk_linear.shape[0], chunk_linear.shape[1], 3).astype(np.float32)) * 0.5
                    noise = noise * st_lw
                    dD = noise * (eff_strength * 0.35 * grain_scale * dens_mod[..., None])
                    chunk_linear = np.maximum(0.0, chunk_linear * np.exp(-dD * 2.3025851))

                coords = np.clip(shaper_forward(chunk_linear) * (lut_size - 1), 0.0, lut_size - 1.0001)
                idx0 = coords.astype(np.int32)
                idx1 = np.minimum(idx0 + 1, lut_size - 1)
                d = coords - idx0

                r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
                r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

                c000 = lut_3d[b0, g0, r0]
                c100 = lut_3d[b0, g0, r1]
                c010 = lut_3d[b0, g1, r0]
                c110 = lut_3d[b0, g1, r1]
                c001 = lut_3d[b1, g0, r0]
                c101 = lut_3d[b1, g0, r1]
                c011 = lut_3d[b1, g1, r0]
                c111 = lut_3d[b1, g1, r1]

                dr = d[..., 0:1]
                dg = d[..., 1:2]
                db = d[..., 2:3]

                c00 = c000 * (1.0 - dr) + c100 * dr
                c01 = c001 * (1.0 - dr) + c101 * dr
                c10 = c010 * (1.0 - dr) + c110 * dr
                c11 = c011 * (1.0 - dr) + c111 * dr

                c0 = c00 * (1.0 - dg) + c10 * dg
                c1 = c01 * (1.0 - dg) + c11 * dg

                chunk_film = c0 * (1.0 - db) + c1 * db

                # DIR Coupler
                if abs(dir_intl - 1.0) > 0.01 or abs(dir_same - 1.0) > 0.01:
                    luma_f = 0.299 * chunk_film[..., 0] + 0.587 * chunk_film[..., 1] + 0.114 * chunk_film[..., 2]
                    chroma = chunk_film - luma_f[..., None]
                    chunk_film = luma_f[..., None] + chroma * dir_intl
                    chunk_film = np.power(np.clip(chunk_film, 0.0001, 1.0), 1.0 / max(0.2, dir_same))

                # Enlarger Color Timing (CMY filters & print exposure)
                chunk_film = np.power(np.clip(chunk_film, 0.0001, 1.0), cmy_factor) * print_exp
                chunk_film += preflash

                # Paper Chemistry Morph & Developer Exhaustion
                if abs(morph_gamma - 1.0) > 0.01:
                    chunk_film = np.power(np.clip(chunk_film, 0.0001, 1.0), morph_gamma)
                if dev_exh > 0.005:
                    chunk_film = chunk_film * (1.0 - dev_exh * 0.6) + (chunk_film * 0.88 + 0.03) * (dev_exh * 0.6)

                rendered_float[y0:y1] = np.clip(chunk_film, 0.0, 1.0)

            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_chunk, ch) for ch in chunks]
                for fut in concurrent.futures.as_completed(futures):
                    if cancel_cb and cancel_cb():
                        executor.shutdown(wait=False, cancel_futures=True)
                        return {"success": False, "cancelled": True, "error": "用户已取消导出"}
                    fut.result()
                    completed += 1
                    if progress_cb:
                        pct = int(35 + (completed / float(len(chunks))) * 55)
                        progress_cb(pct, f"正在进行多核并发高精度暗房印相 ({int(completed / float(len(chunks)) * 100)}%)...")
                    time.sleep(0.001)

            if cancel_cb and cancel_cb():
                return {"success": False, "cancelled": True, "error": "用户已取消导出"}

            if progress_cb:
                progress_cb(92, "正在编码并写入图像文件...")

            if os.path.isdir(output_path) or output_path.endswith(('/', '\\')):
                src = source_path or params_dict.get("source_path") or self.current_file_path or "developed"
                base_name = os.path.splitext(os.path.basename(src))[0]
                fmt_str = str(format_type).lower()
                ext = ".tif" if "tif" in fmt_str else (".png" if "png" in fmt_str else ".jpg")
                output_path = os.path.join(output_path, f"{base_name}_developed{ext}")

            # Apply crop and straighten rotation if enabled in CPU export
            crop_cfg = params_dict.get("crop", {})
            if isinstance(crop_cfg, dict) and crop_cfg.get("enabled", False):
                # 1. Flips
                if crop_cfg.get("flip_h", False):
                    rendered_float = cv2.flip(rendered_float, 1)
                if crop_cfg.get("flip_v", False):
                    rendered_float = cv2.flip(rendered_float, 0)

                # 2. 90-degree step rotation
                rot_90 = int(crop_cfg.get("rot_90", 0)) % 360
                if rot_90 == 90:
                    rendered_float = cv2.rotate(rendered_float, cv2.ROTATE_90_CLOCKWISE)
                elif rot_90 == 180:
                    rendered_float = cv2.rotate(rendered_float, cv2.ROTATE_180)
                elif rot_90 == 270:
                    rendered_float = cv2.rotate(rendered_float, cv2.ROTATE_90_COUNTERCLOCKWISE)

                rH, rW = rendered_float.shape[:2]
                cx = int(max(0, min(rW - 1, float(crop_cfg.get("x", 0.0)) * rW)))
                cy = int(max(0, min(rH - 1, float(crop_cfg.get("y", 0.0)) * rH)))
                cw = int(max(1, min(rW - cx, float(crop_cfg.get("w", 1.0)) * rW)))
                ch = int(max(1, min(rH - cy, float(crop_cfg.get("h", 1.0)) * rH)))
                c_angle = float(crop_cfg.get("angle", 0.0))
                if abs(c_angle) > 0.01:
                    center_pt = (float(rW * 0.5), float(rH * 0.5))
                    M = cv2.getRotationMatrix2D(center_pt, -c_angle, 1.0)
                    rendered_float = cv2.warpAffine(rendered_float, M, (rW, rH), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                rendered_float = np.ascontiguousarray(rendered_float[cy:cy+ch, cx:cx+cw])

            enc_params = dict(params_dict)
            enc_params["format"] = format_type
            enc_params["quality"] = quality
            enc_params["source_path"] = target_src
            save_encoded_image(rendered_float, output_path, enc_params)

            if progress_cb:
                progress_cb(100, "导出完成")

            return {
                "success": True,
                "output_path": output_path,
                "width": rendered_float.shape[1],
                "height": rendered_float.shape[0]
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"导出失败: {str(e)}"}

    render_full_res = export_image

    def apply_lut_to_linear(self, linear_f32, lut_3d, as_float=False):
        """Vectorized trilinear 3D LUT application with log shaper to linear input (H, W, 3) float32."""
        if lut_3d is None or linear_f32 is None:
            return linear_f32
        lut_size = lut_3d.shape[0]
        coords = np.clip(shaper_forward(linear_f32) * (lut_size - 1), 0.0, lut_size - 1.0001)

        idx0 = coords.astype(np.int32)
        idx1 = np.minimum(idx0 + 1, lut_size - 1)
        d = coords - idx0

        r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
        r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

        c000 = lut_3d[b0, g0, r0]
        c100 = lut_3d[b0, g0, r1]
        c010 = lut_3d[b0, g1, r0]
        c110 = lut_3d[b0, g1, r1]
        c001 = lut_3d[b1, g0, r0]
        c101 = lut_3d[b1, g0, r1]
        c011 = lut_3d[b1, g1, r0]
        c111 = lut_3d[b1, g1, r1]

        dr = d[..., 0:1]
        dg = d[..., 1:2]
        db = d[..., 2:3]

        c00 = c000 * (1 - dr) + c100 * dr
        c01 = c001 * (1 - dr) + c101 * dr
        c10 = c010 * (1 - dr) + c110 * dr
        c11 = c011 * (1 - dr) + c111 * dr

        c0 = c00 * (1 - dg) + c10 * dg
        c1 = c01 * (1 - dg) + c11 * dg

        out = c0 * (1 - db) + c1 * db
        if as_float:
            return np.clip(out, 0.0, 1.0)
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

    def apply_lut_to_rgb(self, rgb_u8, lut_3d):
        """Fast vectorized trilinear 3D LUT application to thumbnail (H, W, 3) uint8."""
        if lut_3d is None or rgb_u8 is None:
            return rgb_u8
        lut_size = lut_3d.shape[0]
        coords = (rgb_u8.astype(np.float32) / 255.0) * (lut_size - 1)
        coords = np.clip(coords, 0.0, lut_size - 1.0001)

        idx0 = coords.astype(np.int32)
        idx1 = np.minimum(idx0 + 1, lut_size - 1)
        d = coords - idx0

        r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
        r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

        c000 = lut_3d[b0, g0, r0]
        c100 = lut_3d[b0, g0, r1]
        c010 = lut_3d[b0, g1, r0]
        c110 = lut_3d[b0, g1, r1]
        c001 = lut_3d[b1, g0, r0]
        c101 = lut_3d[b1, g0, r1]
        c011 = lut_3d[b1, g1, r0]
        c111 = lut_3d[b1, g1, r1]

        dr = d[..., 0:1]
        dg = d[..., 1:2]
        db = d[..., 2:3]

        c00 = c000 * (1 - dr) + c100 * dr
        c01 = c001 * (1 - dr) + c101 * dr
        c10 = c010 * (1 - dr) + c110 * dr
        c11 = c011 * (1 - dr) + c111 * dr

        c0 = c00 * (1 - dg) + c10 * dg
        c1 = c01 * (1 - dg) + c11 * dg

        out = c0 * (1 - db) + c1 * db
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

    def export_lut(self, params_dict, output_cube_path, lut_size=33):
        """Bake the current film + print + enlarger settings into a standard 3D .cube LUT."""
        os.makedirs(os.path.dirname(os.path.abspath(output_cube_path)), exist_ok=True)
        
        film_stock = params_dict.get("film_stock", "kodak_portra_400")
        paper_stock = params_dict.get("paper_stock", "kodak_2383")

        lin = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        r, g, b = np.meshgrid(lin, lin, lin, indexing='ij')
        lattice = np.stack([r, g, b], axis=-1).reshape(-1, 1, 3)

        params = init_params()
        params.film = self._load_profile(film_stock)
        is_pos = (params.film and hasattr(params.film, 'is_positive') and params.film.is_positive) or \
                 (params.film and hasattr(params.film, 'info') and getattr(params.film.info, 'type', None) == 'positive')
        is_bw = (params.film and hasattr(params.film, 'info') and getattr(params.film.info, 'channel_model', None) == 'bw')

        if is_pos:
            params.io.scan_film = True
            params.print = None

        has_dir = True
        if is_bw or is_pos:
            has_dir = False
        else:
            phys_ext = getattr(params.film.data, "physical_extensions", None) or {}
            c_ext = phys_ext.get("couplers") if isinstance(phys_ext, dict) else None
            if isinstance(c_ext, dict):
                cinh = c_ext.get("coupler_inhibition")
                cgeom = c_ext.get("coupler_geometry")
                if cinh is not None and np.sum(np.abs(cinh)) < 1e-6 and not cgeom:
                    has_dir = False
        params.film_render.dir_couplers.active = has_dir

        params.debug.lut_mode = True
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = float(params_dict.get("exposure_ev", 0.0))
        if is_bw:
            base_filter = [0.0, 0.0, 0.0]
        else:
            base_filter = self.get_neutral_filter(paper_stock, film_stock)
        params.settings.neutral_print_filters_from_database = False
        params.enlarger.c_filter_neutral = float(base_filter[0])
        params.enlarger.m_filter_neutral = float(base_filter[1])
        params.enlarger.y_filter_neutral = float(base_filter[2])
        params.enlarger.m_filter_shift = float(params_dict.get("enlarger_magenta", 0.0))
        params.enlarger.y_filter_shift = float(params_dict.get("enlarger_yellow", 0.0))
        params.enlarger.preflash_exposure = float(params_dict.get("pre_flash", 0.0))
        params.enlarger.print_exposure = float(params_dict.get("print_exposure", 1.0))
        params.film_render.halation.active = False
        params.film_render.grain.active = False

        digested = digest_params(params)
        simulator = Simulator(digested)
        lut_output = np.clip(simulator.process(lattice).reshape(lut_size, lut_size, lut_size, 3), 0.0, 1.0)
        if is_bw:
            luma = 0.2126 * lut_output[..., 0] + 0.7152 * lut_output[..., 1] + 0.0722 * lut_output[..., 2]
            lut_output[..., 0] = luma
            lut_output[..., 1] = luma
            lut_output[..., 2] = luma
        elif is_pos:
            lut_output = self._postprocess_positive_lut(lut_output, film_stock)

        with open(output_cube_path, "w", encoding="utf-8") as f:
            f.write(f"# Created by SpektraDarkroom Vulkan Edition\n")
            f.write(f"# Film: {film_stock}, Paper: {paper_stock}\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            for b_idx in range(lut_size):
                for g_idx in range(lut_size):
                    for r_idx in range(lut_size):
                        rgb = lut_output[r_idx, g_idx, b_idx]
                        f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

        return {
            "success": True,
            "lut_path": output_cube_path
        }

    def import_film_lut(self, src_path, custom_name="", custom_desc=""):
        return self.import_custom_stock("film", src_path, custom_name, custom_desc)

    def import_paper_lut(self, src_path, custom_name="", custom_desc=""):
        return self.import_custom_stock("paper", src_path, custom_name, custom_desc)

    def import_custom_stock(self, stock_type, src_path, custom_name="", custom_desc=""):
        import json
        dest_dir = self.profiles_dir
        os.makedirs(dest_dir, exist_ok=True)
        fname = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, fname)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if custom_name:
                data["name"] = custom_name
            if custom_desc:
                if "info" in data and isinstance(data["info"], dict):
                    data["info"]["desc"] = custom_desc
                data["desc"] = custom_desc
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            import shutil
            shutil.copy2(src_path, dest_path)

        self._lut_cache.clear()
        self._profile_cache.clear()
        return dest_path

    def delete_custom_stock(self, stock_type, stock_id):
        """Physically deletes user-imported profile from disk and clears caches."""
        dest_dir = self.profiles_dir
        for f in os.listdir(dest_dir):
            if f.endswith(".json") and (f.startswith(stock_id) or stock_id in f):
                fp = os.path.join(dest_dir, f)
                try:
                    os.remove(fp)
                except Exception:
                    pass
        self._lut_cache.clear()
        self._profile_cache.clear()
        return True

