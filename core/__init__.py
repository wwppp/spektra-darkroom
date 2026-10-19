"""
Spektra Darkroom - Core Architecture Package
Provides core processing engines, configuration management, session caching, SDC sidecar handling,
centralized format constants, and path utilities.
"""

import sys

# Submodule imports
from . import version
from . import format_constants
from . import path_utils
from . import config_manager
from . import sdc_manager
from . import session_cache_manager
from . import app_core

# Re-export key symbols for ergonomic access (e.g. from core import SpektraEngine, config_manager)
from .version import (
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
    VERSION_BUILD,
    VERSION_STRING,
    get_version,
    get_app_title,
    get_full_version_info,
)

from .format_constants import (
    RAW_EXTENSIONS,
    STANDARD_IMAGE_EXTENSIONS,
    ALL_SUPPORTED_IMAGE_EXTENSIONS,
    FILE_DIALOG_IMAGE_FILTER,
    get_file_extension,
    is_raw_file,
    is_supported_image,
)

from .path_utils import (
    get_base_dir,
    get_resource_dir,
    get_icon_dir,
    get_icon_path,
    normalize_path,
)

from .app_core import (
    SpektraEngine,
    shaper_forward,
    shaper_inverse,
    prophoto_to_srgb_filmic,
    srgb_to_linear,
    SRGB_TO_PROPHOTO,
    apply_highlights_shadows_np,
    apply_diffusion_np,
    save_encoded_image,
    extract_exif_bundle,
    get_icc_profile_bytes,
)

# Register top-level aliases in sys.modules for flawless backward compatibility
_aliases = {
    "app_core": app_core,
    "config_manager": config_manager,
    "session_cache_manager": session_cache_manager,
    "sdc_manager": sdc_manager,
    "format_constants": format_constants,
    "path_utils": path_utils,
    "version": version,
}
for _name, _mod in _aliases.items():
    sys.modules.setdefault(_name, _mod)

__all__ = [
    # Submodules
    "app_core",
    "config_manager",
    "session_cache_manager",
    "sdc_manager",
    "format_constants",
    "path_utils",
    "version",
    # Versioning
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",
    "VERSION_BUILD",
    "VERSION_STRING",
    "get_version",
    "get_app_title",
    "get_full_version_info",
    # Format constants
    "RAW_EXTENSIONS",
    "STANDARD_IMAGE_EXTENSIONS",
    "ALL_SUPPORTED_IMAGE_EXTENSIONS",
    "FILE_DIALOG_IMAGE_FILTER",
    "get_file_extension",
    "is_raw_file",
    "is_supported_image",
    # Path utils
    "get_base_dir",
    "get_resource_dir",
    "get_icon_dir",
    "get_icon_path",
    "normalize_path",
    # Engine & processing
    "SpektraEngine",
    "shaper_forward",
    "shaper_inverse",
    "prophoto_to_srgb_filmic",
    "srgb_to_linear",
    "SRGB_TO_PROPHOTO",
    "apply_highlights_shadows_np",
    "apply_diffusion_np",
    "save_encoded_image",
    "extract_exif_bundle",
    "get_icc_profile_bytes",
]
