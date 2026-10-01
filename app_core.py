import os
import json
import base64
import time
import io
import threading
import uuid
import numpy as np
import cv2
from PIL import Image

import spektrafilm
from spektrafilm.runtime.api import init_params, digest_params
from spektrafilm.runtime.process import Simulator
from spektrafilm.profiles import load_processed_profile

try:
    import rawpy
except ImportError:
    rawpy = None


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


def srgb_to_linear(srgb):
    """Decode standard sRGB non-linear CCTF into linear values."""
    srgb_c = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb_c <= 0.04045, srgb_c / 12.92, np.power(np.maximum(0.0, (srgb_c + 0.055) / 1.055), 2.4)).astype(np.float32)


def prophoto_to_srgb(lin_pro):
    """Convert Linear ProPhoto RGB to standard Display sRGB."""
    lin_srgb = np.maximum(0.0, lin_pro @ PROPHOTO_TO_SRGB.T)
    srgb_disp = np.where(
        lin_srgb <= 0.0031308,
        12.92 * lin_srgb,
        1.055 * np.power(np.maximum(lin_srgb, 1e-6), 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb_disp, 0.0, 1.0).astype(np.float32)


def filmic_shoulder_np(c, threshold=0.75):
    """Filmic soft highlight roll-off (smooth C1 shoulder compression above threshold)."""
    over = np.maximum(0.0, c - threshold)
    span = 1.0 - threshold
    return (np.minimum(c, threshold) + span * (over / (span + over))).astype(np.float32)


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
    from path_utils import get_resource_dir
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


def extract_exif_bundle(source_path):
    """Extract complete camera metadata from original photo or RAW negative (ARW, CR2, NEF, DNG, RAF, JPG, TIF, PNG).
    Returns (pil_exif, exif_bytes, tiff_tags) for embedding into exported JPEG, TIFF, and PNG images.
    """
    if not source_path or not os.path.exists(source_path):
        return None, None, []

    from PIL import Image
    ext = os.path.splitext(source_path)[1].lower()
    pil_exif = Image.Exif()
    sub_exif = pil_exif.get_ifd(0x8769)
    sub_gps = pil_exif.get_ifd(0x8825)

    SUBTAG_MAP = {
        'ExposureTime': 0x829a, 'FNumber': 0x829d, 'ExposureProgram': 0x8822,
        'ISOSpeedRatings': 0x8827, 'SensitivityType': 0x8830, 'RecommendedExposureIndex': 0x8832,
        'ExifVersion': 0x9000, 'DateTimeOriginal': 0x9003, 'DateTimeDigitized': 0x9004,
        'OffsetTime': 0x9010, 'OffsetTimeOriginal': 0x9011, 'OffsetTimeDigitized': 0x9012,
        'BrightnessValue': 0x9203, 'ExposureBiasValue': 0x9204, 'MaxApertureValue': 0x9205,
        'MeteringMode': 0x9207, 'LightSource': 0x9208, 'Flash': 0x9209,
        'FocalLength': 0x920a, 'MakerNote': 0x927c, 'UserComment': 0x9286,
        'SubsecTime': 0x9290, 'SubsecTimeOriginal': 0x9291, 'SubsecTimeDigitized': 0x9292,
        'FlashpixVersion': 0xa000, 'ColorSpace': 0xa001, 'PixelXDimension': 0xa002, 'PixelYDimension': 0xa003,
        'CustomRendered': 0xa401, 'ExposureMode': 0xa402, 'WhiteBalance': 0xa403,
        'DigitalZoomRatio': 0xa404, 'FocalLengthIn35mmFilm': 0xa405, 'SceneCaptureType': 0xa406,
        'Contrast': 0xa408, 'Saturation': 0xa409, 'Sharpness': 0xa40a,
        'BodySerialNumber': 0xa431, 'LensSpecification': 0xa432, 'LensMake': 0xa433, 'LensModel': 0xa434,
    }

    # 1. Try reading with PIL directly (works for JPG, TIFF, PNG, DNG)
    try:
        with Image.open(source_path) as im:
            loaded_exif = im.getexif()
            if loaded_exif:
                for k, v in loaded_exif.items():
                    try:
                        pil_exif[k] = v
                    except Exception:
                        pass
                if 0x8769 in loaded_exif:
                    for sk, sv in loaded_exif.get_ifd(0x8769).items():
                        try:
                            sub_exif[sk] = sv
                        except Exception:
                            pass
                if 0x8825 in loaded_exif:
                    for gk, gv in loaded_exif.get_ifd(0x8825).items():
                        try:
                            sub_gps[gk] = gv
                        except Exception:
                            pass
    except Exception:
        pass

    # 2. Fuji RAF embedded JPEG preview header
    if ext == '.raf':
        try:
            with open(source_path, 'rb') as rf:
                hdr = rf.read(128)
                if hdr.startswith(b'FUJIFILMCCD-RAW'):
                    import struct
                    off_jpg = struct.unpack('>I', hdr[84:88])[0]
                    len_jpg = struct.unpack('>I', hdr[88:92])[0]
                    rf.seek(off_jpg)
                    jpg_bytes = rf.read(len_jpg)
                    with Image.open(io.BytesIO(jpg_bytes)) as j_img:
                        loaded_exif = j_img.getexif()
                        if loaded_exif:
                            for k, v in loaded_exif.items():
                                try:
                                    pil_exif[k] = v
                                except Exception:
                                    pass
                            if 0x8769 in loaded_exif:
                                for sk, sv in loaded_exif.get_ifd(0x8769).items():
                                    try:
                                        sub_exif[sk] = sv
                                    except Exception:
                                        pass
                            if 0x8825 in loaded_exif:
                                for gk, gv in loaded_exif.get_ifd(0x8825).items():
                                    try:
                                        sub_gps[gk] = gv
                                    except Exception:
                                        pass
        except Exception:
            pass

    # 3. Tifffile for RAW negative formats (ARW, CR2, NEF, DNG, TIFF)
    try:
        import tifffile
        with tifffile.TiffFile(source_path) as tf:
            for page in tf.pages:
                for tag in page.tags:
                    code = tag.code
                    val = tag.value
                    if code in (0x014a, 0x0201, 0x0202, 0x00fe, 0x0103, 0x011a, 0x011b, 0x0128, 0x0100, 0x0101):
                        continue
                    if tag.name == 'ExifTag' and isinstance(val, dict):
                        for sk, sv in val.items():
                            sub_code = sk if isinstance(sk, int) else SUBTAG_MAP.get(sk)
                            if sub_code:
                                try:
                                    sub_exif[sub_code] = sv
                                except Exception:
                                    pass
                    elif isinstance(code, int) and code < 0xffff:
                        try:
                            if code not in pil_exif:
                                pil_exif[code] = val
                        except Exception:
                            pass
    except Exception:
        pass

    # Force orientation to 1 (normal upright) because raw decoding already renders pixels upright
    pil_exif[0x0112] = 1

    exif_bytes = None
    try:
        exif_bytes = pil_exif.tobytes()
    except Exception:
        pass

    tiff_tags = []
    # Tags that tifffile manages internally or should not be written to extratags
    skip_tiff_codes = {
        0x8769, 0x8825, 270, 305, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284
    }
    seen_codes = set()

    def _add_tiff_tag(c, v):
        if c in skip_tiff_codes or c in seen_codes:
            return
        if c == 0x0112:
            v = 1
        seen_codes.add(c)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            tiff_tags.append((c, "I", 1, int(v), True))
        elif hasattr(v, "numerator") and hasattr(v, "denominator"):
            tiff_tags.append((c, "2I", 1, (int(v.numerator), int(v.denominator)), True))
        elif isinstance(v, str):
            tiff_tags.append((c, "s", 0, v, True))
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int):
            tiff_tags.append((c, "2I", 1, (int(v[0]), int(v[1])), True))

    for code, val in pil_exif.items():
        _add_tiff_tag(code, val)

    for code, val in sub_exif.items():
        _add_tiff_tag(code, val)

    return pil_exif, exif_bytes, tiff_tags


class SpektraEngine:
    def __init__(self, resources_dir="resources"):
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
            self._profile_cache[stock_id] = load_processed_profile(stock_id)
        return self._profile_cache[stock_id]

    def get_film_stocks(self):
        return [
            {
                "category": "数码原生 (Bypass)",
                "stocks": [
                    {
                        "id": "none",
                        "name": "无胶卷 (原图直显)",
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

    def get_paper_stocks(self):
        return [
            {
                "id": "none",
                "name": "数码扫描 (Film Scan)",
                "badge": "数码直扫",
                "desc": "跳过相纸印相，呈现底片直接数码扫描的通透细节与宽容度。"
            },
            {
                "id": "kodak_2383",
                "name": "Kodak Vision 2383 放映相纸",
                "badge": "好莱坞标准",
                "desc": "电影放映拷贝标准片，带来深沉厚重的黑位（D-max）与经典的青橙分色立体感。"
            },
            {
                "id": "kodak_2393",
                "name": "Kodak Vision 2393 电影正片",
                "badge": "高密黑阶",
                "desc": "专业电影放映正片，拥有极高的最大光学密度与纯净高光分离感。"
            },
            {
                "id": "kodak_portra_endura",
                "name": "Kodak Portra Endura 经典相纸",
                "badge": "人像展览级",
                "desc": "暗房人像定制相纸，高光平滑过渡，呈现极其优雅柔和的皮肤质感。"
            },
            {
                "id": "kodak_ultra_endura",
                "name": "Kodak Ultra Endura 高反差相纸",
                "badge": "通透锐利",
                "desc": "更高的反差和色彩浓度，适合现代风光与建筑摄影的有力呈现。"
            },
            {
                "id": "kodak_supra_endura",
                "name": "Kodak Supra Endura 专业相纸",
                "badge": "均衡旗舰",
                "desc": "全能型专业暗房相纸，动态范围广阔，色彩还原生动饱和。"
            },
            {
                "id": "kodak_endura_premier",
                "name": "Kodak Endura Premier 顶级相纸",
                "badge": "数码暗房顶峰",
                "desc": "柯达顶级数码暗房相纸，扩展色彩域与高光洁白度。"
            },
            {
                "id": "kodak_ektacolor_edge",
                "name": "Kodak Ektacolor Edge 经典相纸",
                "badge": "平民经典",
                "desc": "柯达广受欢迎的日常彩色冲印相纸，温暖平实的生活感色彩。"
            },
            {
                "id": "fujifilm_crystal_archive_typeii",
                "name": "Fujifilm Crystal Archive II 水晶相纸",
                "badge": "纯净高光",
                "desc": "富士纯白高光表现，色彩鲜艳明快，层次丰富持久。"
            }
        ]

    def get_neutral_filter(self, paper_stock, film_stock):
        if paper_stock in self.neutral_filters:
            paper_dict = self.neutral_filters[paper_stock]
            for illuminant_key in ["TH-KG3", "tungsten", "default"]:
                if illuminant_key in paper_dict and film_stock in paper_dict[illuminant_key]:
                    return list(paper_dict[illuminant_key][film_stock])
        return [0.0, 50.0, 50.0]

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

        eff_paper = "scan" if print_mode == "scan" else paper_stock
        cache_key = (film_stock, eff_paper, lut_size, illuminant, dir_amt, dir_intl, dir_same, morph_gamma, dev_exh)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]

        # 1. Bypass mode: 胶卷为 "none" 时，生成从 Linear ProPhoto RGB -> Display sRGB 的单位转换表 (保持无滤镜自然色彩)
        if film_stock == "none":
            lin = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
            b, g, r = np.meshgrid(lin, lin, lin, indexing='ij')
            lattice = np.stack([r, g, b], axis=-1).astype(np.float32)
            identity_lut = prophoto_to_srgb(lattice)
            self._lut_cache[cache_key] = identity_lut
            return identity_lut

        cache_dir = os.path.join(self.resources_dir, ".lut_cache")
        if is_default_params:
            cache_file = os.path.join(cache_dir, f"v2__{film_stock}__{eff_paper}__{lut_size}.npy")
            if os.path.exists(cache_file):
                try:
                    lut_3d = np.load(cache_file)
                    self._lut_cache[cache_key] = lut_3d
                    return lut_3d
                except Exception:
                    pass

        print(f"[SpektraEngine] 正在为 GPU 生成物理 3D LUT: {film_stock} + {eff_paper} (模式: {print_mode}, 光源: {illuminant}, DIR: {dir_intl}, 尺寸: {lut_size})...")
        t0 = time.time()
        
        lin = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        b, g, r = np.meshgrid(lin, lin, lin, indexing='ij')
        lattice = np.stack([r, g, b], axis=-1)  # RGB channel values
        flat_input = lattice.reshape(-1, 1, 3)

        params = init_params()
        params.film = self._load_profile(film_stock)

        # Check if positive (slide/reversal) or negative film
        is_pos = (params.film and hasattr(params.film, 'is_positive') and params.film.is_positive) or \
                 (params.film and hasattr(params.film, 'info') and getattr(params.film.info, 'type', None) == 'positive')

        if is_pos:
            # Positive slide film: inherently positive, no photographic print paper
            params.io.scan_film = True
            params.print = None
            params.settings.neutral_print_filters_from_database = False
            base_paper = "none"
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

        # DIR Couplers
        params.film_render.dir_couplers.active = (dir_amt > 0.01)
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

        base_filter = self.get_neutral_filter(base_paper, film_stock)
        params.enlarger.filter_cyan = base_filter[0]
        params.enlarger.filter_magenta = base_filter[1]
        params.enlarger.filter_yellow = base_filter[2]
        params.film_render.halation.active = False
        params.film_render.grain.active = False

        digested = digest_params(params)
        sim = Simulator(digested)
        lut_output = sim.process(flat_input)
        lut_3d = np.clip(lut_output.reshape(lut_size, lut_size, lut_size, 3), 0.0, 1.0).astype(np.float32)

        self._lut_cache[cache_key] = lut_3d
        if is_default_params:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_file, lut_3d)
            except Exception:
                pass
        print(f"[SpektraEngine] 3D LUT 生成完毕，耗时: {round((time.time() - t0)*1000, 1)}ms")
        return lut_3d

    def _extract_exif(self, file_path):
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
            "focal_length": "-"
        }
        
        # File Size
        try:
            sz_bytes = os.path.getsize(file_path)
            meta["file_size"] = f"{sz_bytes / (1024.0 * 1024.0):.1f} MB"
        except Exception:
            pass

        ext = os.path.splitext(file_path)[1].lower()

        # Helper mapping tables
        prog_map = {
            0: "未定义", 1: "手动 (M)", 2: "程序自动 (P)", 3: "光圈优先 (A)",
            4: "快门优先 (S)", 5: "创意程序", 6: "动作程序", 7: "人像模式", 8: "风景模式"
        }
        meter_map = {
            0: "未知", 1: "平均测光", 2: "中央重点平均", 3: "点测光",
            4: "多点测光", 5: "多重 / 图案测光", 6: "局部测光"
        }

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
                                if meta["capture_time"] == "-": meta["capture_time"] = str(ex.get(0x0132, "-")).strip()
                                if 0x8769 in ex:
                                    sub = ex.get_ifd(0x8769)
                                    if meta["lens_model"] == "-":
                                        raw_lens = sub.get(0xa434, "-")
                                        meta["lens_model"] = str(raw_lens).strip().strip("\x00").strip()
                                    if meta["lens_make"] == "-":
                                        raw_lmake = sub.get(0xa433, "-")
                                        meta["lens_make"] = str(raw_lmake).strip().strip("\x00").strip()
                                    if meta["serial_number"] == "-":
                                        raw_sn = sub.get(0xa431, "-")
                                        meta["serial_number"] = str(raw_sn).strip().strip("\x00").strip()
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
                                        if fl_val is not None: meta["flash"] = "开启闪光" if (int(fl_val) & 1) else "无闪光"
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
                        elif t_name == 'BodySerialNumber' and meta["serial_number"] == "-":
                            meta["serial_number"] = str(val).strip()
                        elif t_name == 'ExifTag' and isinstance(val, dict):
                            sub = val
                            for k, v in sub.items():
                                if k in ('LensModel', 0xa434) and meta["lens_model"] == "-":
                                    meta["lens_model"] = str(v).strip()
                                elif k in ('LensMake', 0xa433) and meta["lens_make"] == "-":
                                    meta["lens_make"] = str(v).strip()
                                elif k in ('BodySerialNumber', 0xa431) and meta["serial_number"] == "-":
                                    meta["serial_number"] = str(v).strip()
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
                                    meta["flash"] = "开启闪光" if (int(v) & 1) else "无闪光"
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
                    if 0x8769 in exif:
                        sub_exif = exif.get_ifd(0x8769)
                        for k, v in sub_exif.items():
                            t = ExifTags.TAGS.get(k, str(k))
                            if t == "LensModel" and meta["lens_model"] == "-": meta["lens_model"] = str(v).strip()
                            elif t == "LensMake" and meta["lens_make"] == "-": meta["lens_make"] = str(v).strip()
                            elif t == "BodySerialNumber" and meta["serial_number"] == "-": meta["serial_number"] = str(v).strip()
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
                                meta["flash"] = "开启闪光" if (int(v) & 1) else "无闪光"
        except Exception:
            pass

        # 4. Rawpy extraction for white balance, color temp, camera make, model, resolution
        if rawpy is not None and ext in {".arw", ".srf", ".sr2", ".arq", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".raf", ".dng", ".rw2", ".orf", ".ori", ".pef", ".ptx", ".3fr", ".fff", ".x3f"}:
            try:
                with open(file_path, "rb") as fp:
                    with rawpy.imread(fp) as raw:
                        if meta["resolution"] == "-":
                            meta["resolution"] = f"{raw.sizes.width} × {raw.sizes.height}"

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
                                meta["white_balance"] = f"{int(cct_clamped)} K (原片拍摄)"
            except Exception:
                pass

        if "color_temp" not in meta:
            meta["color_temp"] = 5500.0
            meta["tint"] = 1.0
            meta["white_balance"] = "标准 (5500 K)"

        return meta

    def calculate_auto_exposure_ev(self, float_img):
        """Calculate recommended base exposure compensation using author's 18% middle-gray meter."""
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
            # 与原作者 official pipeline 严格保持 100% 一致：输入数据已是线性空间，apply_cctf_decoding=False
            ev = measure_autoexposure_ev(sample_img, color_space='ProPhoto RGB', apply_cctf_decoding=False, method='center_weighted')
            if np.isnan(ev) or np.isinf(ev):
                return 0.0
            return float(np.clip(round(ev, 2), -3.0, 3.0))
        except Exception as e:
            print(f"[SpektraEngine] 测光计算异常: {e}")
            return 0.0

    def load_image(self, file_path):
        """Load RAW or high-fidelity image into float32 array for GPU texture."""
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"文件不存在: {file_path}"}

            ext = os.path.splitext(file_path)[1].lower()
            self.current_file_path = file_path
            self.original_meta = {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "ext": ext
            }
            # Extract rich EXIF shooting attributes
            self.original_meta.update(self._extract_exif(file_path))

            raw_exts = {
                ".arw", ".srf", ".sr2", ".arq",
                ".cr2", ".cr3", ".crw",
                ".nef", ".nrw",
                ".raf",
                ".dng",
                ".rw2",
                ".orf", ".ori",
                ".pef", ".ptx",
                ".3fr", ".fff",
                ".x3f",
            }

            print(f"[SpektraEngine] 正在打开文件: {file_path}")
            if ext in raw_exts:
                if rawpy is None:
                    return {"success": False, "error": "系统未检测到 rawpy 模块，无法解析相机 RAW"}
                
                with open(file_path, "rb") as fp:
                    with rawpy.imread(fp) as raw:
                        try:
                            # 16-bit linear ACES half-size extraction (instant decode in ~200ms)
                            rgb16 = raw.postprocess(
                                use_camera_wb=True,
                                half_size=True,
                                no_auto_bright=True,
                                output_color=rawpy.ColorSpace.ACES,
                                gamma=(1, 1),
                                output_bps=16
                            )
                        except Exception:
                            rgb16 = raw.postprocess(
                                use_auto_wb=True,
                                half_size=True,
                                no_auto_bright=True,
                                output_color=rawpy.ColorSpace.ACES,
                                gamma=(1, 1),
                                output_bps=16
                            )
                post_h, post_w = rgb16.shape[:2]
                w, h = post_w * 2, post_h * 2
                # Convert linear ACES2065-1 to linear ProPhoto RGB
                rgb_aces = rgb16.astype(np.float32) / 65535.0
                rgb_linear_pro = np.maximum(0.0, rgb_aces @ ACES_TO_PROPHOTO.T)
                del rgb16, rgb_aces
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

            self.original_meta["width"] = w
            self.original_meta["height"] = h
            self.original_meta["resolution"] = f"{w} × {h}"

            # Downscale slightly for GPU preview texture according to user preference (1080P/2K/4K)
            cur_h, cur_w = rgb_linear_pro.shape[:2]
            max_edge = 2048
            try:
                import config_manager
                max_edge = int(config_manager.get_preferences().get("preview_max_edge", 2048))
            except Exception:
                max_edge = 2048
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
                # Calculate auto-exposure on Linear ProPhoto RGB image
                auto_ev = self.calculate_auto_exposure_ev(preview_small)
                base_ev = auto_ev
                # Calibrate underlying preview image to 18% middle-gray baseline
                self.raw_preview = (preview_small * float(np.exp2(base_ev))).astype(np.float32)
            else:
                auto_ev = 0.0
                base_ev = 0.0
                self.raw_preview = preview_small.astype(np.float32)
            del preview_small

            # Small filmstrip negative thumbnail (height 72px)
            th_h = 72
            th_w = max(48, int(self.raw_preview.shape[1] * (th_h / float(self.raw_preview.shape[0]))))
            th_img = cv2.resize(self.raw_preview, (th_w, th_h), interpolation=cv2.INTER_AREA)
            th_u8 = (prophoto_to_srgb(th_img) * 255.0).astype(np.uint8)

            # Store in session photos
            existing = next((p for p in self.session_photos if p["path"] == file_path), None)
            photo_id = existing["id"] if existing else f"film_{uuid.uuid4().hex[:8]}"

            photo_item = {
                "id": photo_id,
                "path": file_path,
                "filename": self.original_meta["filename"],
                "width": w,
                "height": h,
                "thumbnail_rgb": th_u8,
                "raw_preview": self.raw_preview,
                "meta": dict(self.original_meta),
                "is_raw": is_raw_file,
                "base_ev": base_ev,
                "auto_ev": 0.0
            }

            if existing:
                self.session_photos[self.session_photos.index(existing)] = photo_item
            else:
                self.session_photos.append(photo_item)

            self.active_photo_id = photo_id

            print(f"[SpektraEngine] 成功载入 {self.original_meta['filename']}: 原始尺寸 {w}x{h}，GPU 纹理尺寸 {self.raw_preview.shape[1]}x{self.raw_preview.shape[0]}，基准测光: {auto_ev:+.2f} EV")
            return {
                "success": True,
                "id": photo_id,
                "filename": self.original_meta["filename"],
                "width": w,
                "height": h,
                "preview_w": self.raw_preview.shape[1],
                "preview_h": self.raw_preview.shape[0],
                "thumbnail_rgb": th_u8,
                "float_img": self.raw_preview,
                "raw_preview": self.raw_preview,
                "exif": dict(self.original_meta),
                "photos": self.get_session_photos(),
                "active_id": photo_id,
                "auto_ev": auto_ev,
                "base_ev": base_ev
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"解析图片或 RAW 发生错误: {str(e)}"}

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
        self.raw_preview = photo["raw_preview"]
        self.raw_full = None
        self.original_meta = dict(photo["meta"])
        self.current_file_path = photo["path"]

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
        raw_exts = {".arw", ".srf", ".sr2", ".arq", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".raf", ".dng", ".rw2", ".orf", ".ori", ".pef", ".ptx", ".3fr", ".fff", ".x3f"}
        if ext in raw_exts:
            with open(file_path, "rb") as fp:
                with rawpy.imread(fp) as raw:
                    try:
                        rgb16 = raw.postprocess(
                            use_camera_wb=True,
                            half_size=False,
                            no_auto_bright=True,
                            output_color=rawpy.ColorSpace.ACES,
                            gamma=(1, 1),
                            output_bps=16
                        )
                    except Exception:
                        rgb16 = raw.postprocess(
                            use_auto_wb=True,
                            half_size=False,
                            no_auto_bright=True,
                            output_color=rawpy.ColorSpace.ACES,
                            gamma=(1, 1),
                            output_bps=16
                        )
                    rgb_aces = rgb16.astype(np.float32) / 65535.0
                    rgb_pro = np.maximum(0.0, rgb_aces @ ACES_TO_PROPHOTO.T)
                    if abs(base_ev) > 0.0001:
                        rgb_pro *= float(np.exp2(base_ev))
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
            c_val = float(params_dict.get("enlarger_cyan", 0.0))
            m_val = float(params_dict.get("enlarger_magenta", 0.0))
            y_val = float(params_dict.get("enlarger_yellow", 0.0))
            print_exp = float(params_dict.get("print_exposure", 1.0))
            preflash = float(params_dict.get("pre_flash", 0.0))
            halation = float(params_dict.get("halation", 0.5))
            hal_bounces = int(params_dict.get("halation_bounces", 2))
            hal_decay = float(params_dict.get("halation_decay", 0.5))
            hal_boost = float(params_dict.get("halation_boost", 0.0))
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
            chunk_size = 512

            base_temp = float(params_dict.get("base_temp", 5500.0))
            base_tint = float(params_dict.get("base_tint", 1.0))
            temp_factor = (temp_val - base_temp) / 3000.0
            tint_factor = tint_val / max(0.001, base_tint)
            cmy_arr = np.array([c_val, m_val, y_val], dtype=np.float32)
            cmy_factor = 1.0 - cmy_arr * 0.015

            for y0 in range(0, H, chunk_size):
                if cancel_cb and cancel_cb():
                    return {"success": False, "cancelled": True, "error": "用户已取消导出"}
                y1 = min(y0 + chunk_size, H)
                if progress_cb:
                    pct = int(10 + (y0 / float(H)) * 80)
                    progress_cb(pct, f"正在进行高精度冲印显影 ({int((y0 / float(H)) * 100)}%)...")

                chunk_orig = raw_target[y0:y1].copy()

                # 1. Exposure compensation
                chunk_linear = chunk_orig * float(np.exp2(exp_val))

                # 2. White balance / temp / tint
                chunk_linear[..., 0] *= (1.0 + temp_factor * 0.22)
                chunk_linear[..., 2] *= (1.0 - temp_factor * 0.22)
                chunk_linear[..., 1] *= tint_factor

                # Optical Diffusion Bloom
                if diff_family != "none" and diff_strength > 0.001:
                    thr = 0.38 if diff_family == "black_pro_mist" else 0.28
                    luma_in = 0.2126 * chunk_linear[..., 0] + 0.7152 * chunk_linear[..., 1] + 0.0722 * chunk_linear[..., 2]
                    hl = np.maximum(0.0, luma_in - thr)[..., None]
                    hl_glow = chunk_linear * hl
                    sigma = max(1.0, 16.0 * diff_strength)
                    if diff_family == "cinebloom": sigma *= 1.8
                    elif diff_family == "glimmerglass": sigma *= 0.85
                    elif diff_family == "pro_mist": sigma *= 1.4
                    ksize = int(sigma * 3.0) | 1
                    try:
                        bloom = cv2.GaussianBlur(hl_glow, (ksize, ksize), sigma)
                    except Exception:
                        bloom = hl_glow
                    b_tint = np.array([1.0 + diff_warmth * 0.35, 1.0 + diff_warmth * 0.08, 1.0 - diff_warmth * 0.35], dtype=np.float32)
                    intensity = 0.85 if diff_family == "pro_mist" else (0.95 if diff_family == "cinebloom" else 0.70)
                    chunk_linear += bloom * (diff_strength * intensity) * b_tint
                    if diff_family == "pro_mist":
                        chunk_linear = chunk_linear * 0.985 + 0.015 * diff_strength

                # 3. 3D LUT sampling (Film emulsion + Paper D-max)
                # Apply filmic highlight shoulder roll-off to avoid hard highlight clipping
                shoulder_linear = filmic_shoulder_np(chunk_linear, 0.75)
                coords = np.clip(shoulder_linear * (lut_size - 1), 0.0, lut_size - 1.0001)
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

                # 4. Enlarger Color Timing
                chunk_film = np.power(np.clip(chunk_film, 0.0001, 1.0), cmy_factor) * print_exp
                chunk_film += preflash

                # 5. Multi-bounce Halation
                if halation > 0.01:
                    luma_orig = 0.299 * chunk_orig[..., 0] + 0.587 * chunk_orig[..., 1] + 0.114 * chunk_orig[..., 2]
                    hl_base = np.maximum(0.0, luma_orig - 0.70 + hal_boost * 0.10) / 0.30
                    hal_accum = np.zeros_like(hl_base)
                    cur_dec = 1.0
                    for b in range(min(4, hal_bounces)):
                        hal_accum += np.power(hl_base, 1.6 + b * 0.4) * cur_dec
                        cur_dec *= hal_decay
                    hal_glow = (hal_accum * halation * 0.35)[..., None]
                    chunk_film[..., 0] += hal_glow[..., 0] * 1.0
                    chunk_film[..., 1] += hal_glow[..., 0] * 0.14
                    chunk_film[..., 2] += hal_glow[..., 0] * 0.04

                # 6. Silver Halide Organic Film Grain (Pure Luminance Density Grain, Zero Chroma Noise)
                if grain > 0.01:
                    luma_film = 0.299 * chunk_film[..., 0] + 0.587 * chunk_film[..., 1] + 0.114 * chunk_film[..., 2]
                    density = 1.0 - np.clip(luma_film, 0.0, 1.0)
                    grain_mask = np.sqrt(np.maximum(0.0, density * (1.0 - density))) * 2.0
                    # Single-channel 2D luminance noise broadcasted across RGB to avoid digital color confetti
                    noise = (np.random.rand(chunk_film.shape[0], chunk_film.shape[1], 1).astype(np.float32) - 0.5) * (grain * grain_scale * 0.14)
                    chunk_film += noise * grain_mask[..., None]

                rendered_float[y0:y1] = np.clip(chunk_film, 0.0, 1.0)

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

            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(output_path):
                try:
                    import stat
                    os.chmod(output_path, stat.S_IWRITE)
                except Exception:
                    pass

            from PIL import Image

            bit_depth = int(params_dict.get("bit_depth", 8))
            dpi_val = int(params_dict.get("dpi", 300))
            fmt_lower = str(format_type).lower()

            colorspace = params_dict.get("colorspace", "sRGB")
            rendered_float = convert_image_colorspace(rendered_float, colorspace)
            icc_bytes = get_icc_profile_bytes(colorspace)

            # Issue 1: Extract complete camera EXIF metadata from original photo or RAW file
            pil_exif, exif_bytes, tiff_tags = extract_exif_bundle(target_src)

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
                    # Attempt desired compression; fall back gracefully to deflate or uncompressed if codec missing
                    for attempt in [comp_val, "deflate", None]:
                        try:
                            tifffile.imwrite(path, data, compression=attempt, resolution=(dpi_val, dpi_val, 'INCH'), photometric='rgb', extratags=extratags if extratags else None)
                            return
                        except Exception:
                            continue
                    tifffile.imwrite(path, data, resolution=(dpi_val, dpi_val, 'INCH'), photometric='rgb', extratags=extratags if extratags else None)

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
                    cv2.imwrite(output_path, cv2.cvtColor(out_u16, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, png_level])
                    # Inject EXIF and ICC into 16-bit PNG
                    try:
                        with open(output_path, 'rb') as pf:
                            pdata = pf.read()
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
                                with open(output_path, 'wb') as pf:
                                    pf.write(pdata[:insert_pos] + b''.join(injected_chunks) + pdata[insert_pos:])
                    except Exception as ex_png:
                        print(f"[SpektraEngine] Warning: PNG 16-bit metadata injection failed: {ex_png}")
                else:
                    out_u8 = (np.clip(rendered_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                    im = Image.fromarray(out_u8, mode='RGB')
                    save_kw = {'compress_level': png_level, 'dpi': (dpi_val, dpi_val)}
                    if icc_bytes:
                        save_kw['icc_profile'] = icc_bytes
                    if exif_bytes:
                        save_kw['exif'] = exif_bytes
                    im.save(output_path, 'PNG', **save_kw)

            else:  # jpeg
                q_scale = int(quality) if quality is not None else int(params_dict.get("quality", 9))
                q_val = 50 + q_scale * 5 if q_scale <= 10 else 95
                q_val = max(10, min(100, q_val))
                
                sub_opt = str(params_dict.get("subsampling", "4:4:4"))
                sub_val = 0 if sub_opt == "4:4:4" else (1 if sub_opt == "4:2:2" else 2)
                prog_val = bool(params_dict.get("progressive", True))

                out_u8 = (rendered_float * 255.0).astype(np.uint8)
                scale_pct = int(params_dict.get("scale_pct", 100))
                if scale_pct < 100 and scale_pct > 0:
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
                im.save(output_path, 'JPEG', **save_kw)

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

    def apply_lut_to_linear(self, linear_f32, lut_3d):
        """Vectorized trilinear 3D LUT application to linear input (H, W, 3) float32 in [0, 1]."""
        if lut_3d is None or linear_f32 is None:
            return linear_f32
        lut_size = lut_3d.shape[0]
        coords = np.clip(linear_f32, 0.0, 1.0) * (lut_size - 1)
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
        params.print = self._load_profile(paper_stock)
        if params.film and hasattr(params.film, 'info') and getattr(params.film.info, 'type', None) == 'positive':
            params.io.scan_film = True
        params.camera.exposure_compensation_ev = float(params_dict.get("exposure_ev", 0.0))
        base_filter = self.get_neutral_filter(paper_stock, film_stock)
        params.enlarger.filter_cyan = max(0.0, base_filter[0] + float(params_dict.get("enlarger_cyan", 0.0)))
        params.enlarger.filter_magenta = max(0.0, base_filter[1] + float(params_dict.get("enlarger_magenta", 0.0)))
        params.enlarger.filter_yellow = max(0.0, base_filter[2] + float(params_dict.get("enlarger_yellow", 0.0)))
        params.enlarger.pre_flash = float(params_dict.get("pre_flash", 0.0))
        params.enlarger.exposure_time = float(params_dict.get("print_exposure", 1.0))
        params.film_render.halation.active = False
        params.film_render.grain.active = False

        digested = digest_params(params)
        simulator = Simulator(digested)
        lut_output = np.clip(simulator.process(lattice).reshape(lut_size, lut_size, lut_size, 3), 0.0, 1.0)

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

    def import_film_lut(self, src_path):
        import shutil
        dest_dir = self.profiles_dir
        os.makedirs(dest_dir, exist_ok=True)
        fname = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, fname)
        shutil.copy2(src_path, dest_path)
        self._lut_cache.clear()
        self._profile_cache.clear()
        return dest_path

    def import_paper_lut(self, src_path):
        import shutil
        dest_dir = self.profiles_dir
        os.makedirs(dest_dir, exist_ok=True)
        fname = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, fname)
        shutil.copy2(src_path, dest_path)
        self._lut_cache.clear()
        self._profile_cache.clear()
        return dest_path
