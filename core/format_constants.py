"""SpektraDarkroom Centralized File Format Definitions.
Single source of truth for all camera RAW negative extensions, standard raster image
extensions, and file dialog filters across the entire application.

Import policy (v0.5.14)
-----------------------
Only camera RAW and *lossless* raster containers are importable:

  * ``RAW_EXTENSIONS``            - camera negatives, full radiance, always accepted.
  * ``LOSSLESS_IMAGE_EXTENSIONS`` - PNG / TIFF (8 or 16 bit integer, 32 bit float in TIFF).
  * ``LOSSY_IMAGE_EXTENSIONS``    - JPEG / WebP, recognised but deliberately *not* importable.

Lossy sources are silently dropped at every import entry point instead of raising a
dialog: an 8-bit JPEG has already discarded the shadow/highlight headroom the darkroom
grades against, and re-encoding it here cannot recover it. They remain listed in
``STANDARD_IMAGE_EXTENSIONS`` so identification / diagnostics can still classify a file,
but nothing routes them into the pipeline.
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

# Irreversible / lossy compression. Recognised, never imported.
LOSSY_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".jpe",
    ".webp"
}

# Lossless raster containers accepted as digital inputs. Bit depth is not encoded in the
# extension (and must not be faked by filtering on it): PNG tops out at 16 bit integer,
# TIFF also carries 32 bit float. The decoder decides how to scale by inspecting the array.
LOSSLESS_IMAGE_EXTENSIONS: Set[str] = {
    ".png",
    ".tif", ".tiff"
}

# Every raster extension the decoder can *identify* (lossy included).
STANDARD_IMAGE_EXTENSIONS: Set[str] = LOSSLESS_IMAGE_EXTENSIONS | LOSSY_IMAGE_EXTENSIONS

# The only set any import path may use.
IMPORTABLE_IMAGE_EXTENSIONS: Set[str] = RAW_EXTENSIONS | LOSSLESS_IMAGE_EXTENSIONS

# Standardized QFileDialog filter strings (lossy formats are not offered)
_RAW_PATTERNS = " ".join(f"*{ext}" for ext in sorted(RAW_EXTENSIONS))
_STD_PATTERNS = " ".join(f"*{ext}" for ext in sorted(LOSSLESS_IMAGE_EXTENSIONS))
_IMPORT_PATTERNS = " ".join(f"*{ext}" for ext in sorted(IMPORTABLE_IMAGE_EXTENSIONS))

FILE_DIALOG_IMAGE_FILTER: str = (
    f"所有支持的图像与 RAW 负片 ({_IMPORT_PATTERNS});;"
    f"相机 RAW 负片 ({_RAW_PATTERNS});;"
    f"无损位图 ({_STD_PATTERNS});;"
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


def is_lossy_image(path_or_filename: str) -> bool:
    """Check if the file uses a lossy container (JPEG / WebP) that must not be imported."""
    ext = get_file_extension(path_or_filename)
    return ext in LOSSY_IMAGE_EXTENSIONS


def is_importable_image(path_or_filename: str) -> bool:
    """Check if the file may enter the pipeline: camera RAW or lossless raster."""
    ext = get_file_extension(path_or_filename)
    return ext in IMPORTABLE_IMAGE_EXTENSIONS


def is_supported_image(path_or_filename: str) -> bool:
    """Alias of :func:`is_importable_image`, kept for callers that gate on importability."""
    return is_importable_image(path_or_filename)
