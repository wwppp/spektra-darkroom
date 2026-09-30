"""SpektraDarkroom Version Configuration
"""

APP_NAME = "SpektraDarkroom"
VERSION_MAJOR = 1
VERSION_MINOR = 3
VERSION_PATCH = 0
BUILD_NUMBER = 30
RELEASE_DATE = "2026-09-30"

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
