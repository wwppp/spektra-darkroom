# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all

ROOT_DIR = os.path.abspath(SPECPATH if 'SPECPATH' in globals() else '.')

block_cipher = None

# Collect complete modules
datas_spektra, bin_spektra, hid_spektra = collect_all('spektrafilm')
datas_colour, bin_colour, hid_colour = collect_all('colour')
datas_rawpy, bin_rawpy, hid_rawpy = collect_all('rawpy')
datas_gl, bin_gl, hid_gl = collect_all('OpenGL')

# Strip raw .py source files from datas so core, ui, and spektrafilm are compiled into PYZ bytecode
datas_spektra = [
    (src, dst) for (src, dst) in datas_spektra
    if not str(src).lower().endswith(('.py', '.pyc', '.pyo'))
]

added_datas = [
    (os.path.join(ROOT_DIR, 'resources'), 'resources'),
    (os.path.join(ROOT_DIR, 'spektrafilm', 'data'), os.path.join('spektrafilm', 'data')),
] + datas_spektra + datas_colour + datas_rawpy + datas_gl

sample_arw = os.path.join(ROOT_DIR, 'sample.ARW')
if os.path.exists(sample_arw):
    added_datas.append((sample_arw, '.'))

added_binaries = bin_spektra + bin_colour + bin_rawpy + bin_gl
added_hidden = [
    'core',
    'core.path_utils',
    'core.app_core',
    'core.raw_border',
    'core.config_manager',
    'core.session_cache_manager',
    'core.sdc_manager',
    'core.format_constants',
    'core.darkroom_math',
    'core.lens_correction',
    'core.colorspace',
    'core.version',
    'ui',
    'ui.about_dialog',
    'ui.accordion',
    'ui.adobe_scrub_slider',
    'ui.canvas_rating_widget',
    'ui.canvas_viewport',
    'ui.confirm_dialog',
    'ui.crop_overlay',
    'ui.delete_photo_dialog',
    'ui.export_dialog',
    'ui.export_queue_manager',
    'ui.fading_stacked_widget',
    'ui.filmstrip_widget',
    'ui.film_filter_header',
    'ui.histogram_widget',
    'ui.main_window',
    'ui.preferences_dialog',
    'ui.relink_dialog',
    'ui.smooth_scroll',
    'ui.spinner_widget',
    'ui.theme_manager',
    'ui.thumbnail_controller',
    'ui.tooltip',
    'ui.unsaved_session_dialog',
    'ui.window_utils',
    'PySide6.QtOpenGLWidgets',
    'PySide6.QtOpenGL',
    'PySide6.QtNetwork',
    'PySide6.QtSvg',
    'OpenGL.platform.win32',
    'scipy.ndimage',
    'scipy.spatial',
    'scipy.interpolate',
    'scipy.optimize',
    'numba',
    'llvmlite',
    'cv2',
    'rawpy',
    'PIL',
    'tifffile',
    'imagecodecs',
] + hid_spektra + hid_colour + hid_rawpy + hid_gl

a = Analysis(
    [os.path.join(ROOT_DIR, 'main.py')],
    pathex=[ROOT_DIR],
    binaries=added_binaries,
    datas=added_datas,
    hiddenimports=list(set(added_hidden)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest',
              # Optional GPU backend for the diffusion filter. Pulling it in
              # adds ~780 MB (cupy + cuFFT/cuRAND/NVRTC) and only helps on
              # CUDA machines, so the shipped build stays CPU-only. The code
              # detects the missing module and falls back automatically.
              'cupy', 'cupyx', 'cuda', 'nvidia'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Dead weight
# ---------------------------------------------------------------------------
# ``collect_all('colour')`` copies the whole wheel into the bundle and a few
# PyInstaller hooks drag in things the app can never reach.  They are dropped
# here, at analysis time, rather than deleted from dist/ afterwards, so a plain
# ``pyinstaller SpektraDarkroom.spec`` stays reproducible.
#
# Entries in a TOC are (destination_name, source_path, typecode), so matching on
# element 0 is matching on the path inside the bundle.
PRUNE_PREFIXES = (
    # 74 MB coverage HTML report that ships inside the colour-science wheel.
    'colour/htmlcov',
    # ~0.3 MB of documented example scripts; nothing imports them.
    'colour/examples',
    # 6 MB of Qt's own translation catalogs; the UI carries its own strings and
    # never constructs a QTranslator.
    'PySide6/translations',
    # AVIF codec. Pillow's Image.init() skips plugins that fail to import, and
    # nothing in the app reads or writes AVIF.
    'PIL/_avif',
)
PRUNE_FILES = (
    # 29 MB FFmpeg backend for cv2.VideoCapture/VideoWriter; the app has no video
    # I/O at all, only imread/imwrite.
    'cv2/opencv_videoio_ffmpeg500_64.dll',
)


def _is_dead_weight(path):
    if path in PRUNE_FILES:
        return True
    return any(path == p or path.startswith(p + '/') for p in PRUNE_PREFIXES)


def _prune_toc(toc, label):
    kept = []
    dropped = []
    for entry in toc:
        dest = str(entry[0]).replace('\\', '/')
        # Extension modules are stored under their dotted module name ('PIL._avif'),
        # data and libraries under their real path ('colour/htmlcov').  Test the path
        # form of both so one rule covers either.
        if _is_dead_weight(dest) or _is_dead_weight(dest.replace('.', '/')):
            dropped.append(entry)
        else:
            kept.append(entry)
    if dropped:
        total = 0
        for entry in dropped:
            try:
                src = entry[1]
                total += os.path.getsize(src) if os.path.isfile(src) else 0
            except OSError:
                pass
        print('[prune] %s: %d entries, %.1f MB' % (label, len(dropped), total / 2 ** 20))
    return kept


a.datas = _prune_toc(a.datas, 'datas')
a.binaries = _prune_toc(a.binaries, 'binaries')

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

app_ico = os.path.join(ROOT_DIR, 'resources', 'app_icon.ico')
if not os.path.exists(app_ico):
    app_ico = os.path.join(ROOT_DIR, 'resources', 'app.ico')

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpektraDarkroom',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=app_ico if os.path.exists(app_ico) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SpektraDarkroom',
)
