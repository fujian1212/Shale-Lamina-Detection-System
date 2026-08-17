# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('license.key', '.')]
binaries = []
hiddenimports = ['rock_core_analyzer', 'rock_core_analyzer.core', 'rock_core_analyzer.core.detector', 'rock_core_analyzer.core.image_io', 'rock_core_analyzer.core.preprocessing', 'rock_core_analyzer.core.detection', 'rock_core_analyzer.core.alignment', 'rock_core_analyzer.core.statistics', 'rock_core_analyzer.core.visualization', 'rock_core_analyzer.core.paper_export', 'rock_core_analyzer.core.export', 'rock_core_analyzer.batch', 'rock_core_analyzer.batch.processing', 'rock_core_analyzer.batch.merge', 'rock_core_analyzer.batch.batch_viz', 'rock_core_analyzer.gui', 'rock_core_analyzer.gui.app', 'rock_core_analyzer.gui.ui_setup', 'rock_core_analyzer.gui.single_image', 'rock_core_analyzer.gui.scan_lines', 'rock_core_analyzer.gui.export_ui', 'rock_core_analyzer.gui.batch_ui', 'rock_core_analyzer.gui.workers', 'rock_core_analyzer.gui.utils', 'PIL', 'PIL._tkinter_finder', 'openpyxl', 'scipy', 'scipy.ndimage', 'pandas', 'psutil']
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ttkthemes')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RockCoreAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RockCoreAnalyzer',
)
