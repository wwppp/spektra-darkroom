"""SpektraDarkroom Version Configuration
规范：每次小修复/优化后，自动修改版本号，末位 Patch 递增 1 位（例如 v0.1.0 -> v0.1.1 -> v0.1.2）。
"""

APP_NAME = "SpektraDarkroom"
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 31
BUILD_NUMBER = 32
RELEASE_DATE = "2026-10-02"

VERSION_STRING = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
FULL_VERSION_STRING = f"v{VERSION_STRING} (Build {BUILD_NUMBER})"


def get_app_title():
    return f"{APP_NAME} v{VERSION_STRING}"


def get_full_version_info():
    return {
        "app_name": APP_NAME,
        "version": VERSION_STRING,
        "build": BUILD_NUMBER,
        "release_date": RELEASE_DATE,
        "title": get_app_title(),
        "engine": "Vulkan / GPU Real-Time Hardware Accelerated Pipeline"
    }
