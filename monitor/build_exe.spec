# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Meta-Fingerprint Monitor
# Build: pyinstaller build_exe.spec --clean

import sys
from pathlib import Path

ROOT = Path(SPECPATH)
REPO_SRC_CANDIDATES = [
    ROOT.parent / "src",
    ROOT.parent / "meta_fingerprint_repo" / "meta_fingerprint_repo" / "src",
]
REPO_CONFIG_CANDIDATES = [
    ROOT.parent / "configs",
    ROOT.parent / "meta_fingerprint_repo" / "meta_fingerprint_repo" / "configs",
]
PATHEX = [str(ROOT)] + [str(p) for p in REPO_SRC_CANDIDATES if p.exists()]
DATAS = []
for cfg_dir in REPO_CONFIG_CANDIDATES:
    if cfg_dir.exists():
        DATAS.append((str(cfg_dir), "configs"))
        break

block_cipher = None

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=PATHEX,
    binaries=[],
    datas=DATAS,
    hiddenimports=[
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "pyqtgraph",
        "pyqtgraph.graphicsItems",
        "numpy",
        "scipy",
        "scipy.signal",
        "scipy.stats",
        "reportlab",
        "reportlab.lib",
        "reportlab.platypus",
        "reportlab.graphics",
        "sqlite3",
        "json",
        "socket",
        "threading",
        "struct",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "IPython", "jupyter", "pytest", "tkinter",
        "PySide6", "PyQt6", "wx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MetaFingerprintMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,           # add .ico path here if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MetaFingerprintMonitor",
)
