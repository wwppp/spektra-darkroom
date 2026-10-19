"""SpektraDarkroom Centralized File Format Definitions.
Single source of truth for all camera RAW negative extensions, standard raster image
extensions, and file dialog filters across the entire application.
"""

import os
from typing import Set

# Complete set of camera RAW file extensions natively supported by LibRaw 0.22.1 / rawpy 0.27.1
RAW_EXTENSIONS: Set[str] = {
    # Sony
    ".arw", ".srf", ".sr2", ".arq",
    # Canon
    ".cr2", ".cr3", ".crw",
    # Nikon
    ".nef", ".nrw",
    # Fujifilm
    ".raf",
    # Olympus / OM System (ORF: Olympus Raw Format, ORI: Olympus Raw Image)
    ".orf", ".ori",
    # Panasonic / Lumix / Leica
    ".rw2", ".raw", ".rwl",
    # Pentax / Ricoh
    ".pef", ".ptx",
    # Hasselblad
    ".3fr", ".fff",
    # Phase One / Leaf / Mamiya
    ".iiq", ".cap", ".eip", ".mos", ".mef",
    # Sigma Foveon
    ".x3f",
    # Samsung
    ".srw",
    # Kodak
    ".dcr", ".kdc", ".k25",
    # Minolta / Konica Minolta
    ".mrw",
    # Epson
    ".erf",
    # GoPro
    ".gpr",
    # Casio / Bayer RAW
    ".bay",
    # Adobe Digital Negative (Leica, Apple ProRAW, DJI, Google Pixel, Ricoh GR, etc.)
    ".dng"
}

# Standard raster / SDR developed image file extensions
STANDARD_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".jpe",
    ".png",
    ".tif", ".tiff",
    ".webp"
}

# Complete set of all supported camera negatives and image files
ALL_SUPPORTED_IMAGE_EXTENSIONS: Set[str] = RAW_EXTENSIONS | STANDARD_IMAGE_EXTENSIONS

# Standardized QFileDialog filter strings
_RAW_PATTERNS = " ".join(f"*{ext}" for ext in sorted(RAW_EXTENSIONS))
_STD_PATTERNS = " ".join(f"*{ext}" for ext in sorted(STANDARD_IMAGE_EXTENSIONS))
_ALL_PATTERNS = " ".join(f"*{ext}" for ext in sorted(ALL_SUPPORTED_IMAGE_EXTENSIONS))

FILE_DIALOG_IMAGE_FILTER: str = (
    f"所有支持的图像与 RAW 负片 ({_ALL_PATTERNS});;"
    f"相机 RAW 负片 ({_RAW_PATTERNS});;"
    f"标准图像文件 ({_STD_PATTERNS});;"
    "所有文件 (*.*)"
)


def get_file_extension(path_or_filename: str) -> str:
    """Extract normalized lowercase file extension including dot."""
    if not path_or_filename:
        return ""
    return os.path.splitext(path_or_filename)[1].lower()


def is_raw_file(path_or_filename: str) -> bool:
    """Check if the given file or path is a supported camera RAW negative."""
    ext = get_file_extension(path_or_filename)
    return ext in RAW_EXTENSIONS


def is_supported_image(path_or_filename: str) -> bool:
    """Check if the given file or path is any supported RAW negative or raster image."""
    ext = get_file_extension(path_or_filename)
    return ext in ALL_SUPPORTED_IMAGE_EXTENSIONS
