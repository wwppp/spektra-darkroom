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

added_datas = [
    (os.path.join(ROOT_DIR, 'resources'), 'resources'),
    (os.path.join(ROOT_DIR, 'core'), 'core'),
] + datas_spektra + datas_colour + datas_rawpy + datas_gl

sample_arw = os.path.join(ROOT_DIR, 'sample.ARW')
if os.path.exists(sample_arw):
    added_datas.append((sample_arw, '.'))

added_binaries = bin_spektra + bin_colour + bin_rawpy + bin_gl
added_hidden = [
    'core',
    'core.path_utils',
    'core.app_core',
    'core.config_manager',
    'core.session_cache_manager',
    'core.sdc_manager',
    'core.format_constants',
    'core.version',
    'ui',
    'ui.main_window',
    'ui.adobe_scrub_slider',
    'ui.accordion',
    'ui.canvas_viewport',
    'ui.filmstrip_widget',
    'ui.spinner_widget',
    'ui.export_dialog',
    'PySide6.QtOpenGLWidgets',
    'PySide6.QtOpenGL',
    'OpenGL.platform.win32',
    'scipy.ndimage',
    'scipy.spatial',
    'numba',
    'llvmlite',
    'cv2',
    'rawpy',
    'PIL',
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
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
