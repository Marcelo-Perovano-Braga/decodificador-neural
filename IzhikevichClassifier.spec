# -*- mode: python ; coding: utf-8 -*-

import struct
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH).resolve()
checkpoint_path = project_root / "weights" / "classifier.pt"
icon_path = project_root / "assets" / "app_icon.ico"
version_path = project_root / "packaging" / "windows_version_info.txt"

if sys.platform != "win32":
    raise SystemExit("A especificação deve ser executada no Windows.")
if struct.calcsize("P") * 8 != 64:
    raise SystemExit("Utilize uma instalação Python de 64 bits para gerar o aplicativo.")

if not checkpoint_path.is_file():
    raise SystemExit(
        "Checkpoint ausente: execute train_pipeline.py ou copie "
        "weights/classifier.pt antes de gerar o executável."
    )

for required_path in (icon_path, version_path):
    if not required_path.is_file():
        raise SystemExit(f"Arquivo necessário ausente: {required_path}")

fluent_datas, fluent_binaries, fluent_hidden_imports = collect_all(
    "qfluentwidgets"
)

datas = [
    (str(project_root / "assets"), "assets"),
    (str(checkpoint_path), "weights"),
    *fluent_datas,
]

analysis = Analysis(
    [str(project_root / "main_app.py")],
    pathex=[str(project_root)],
    binaries=fluent_binaries,
    datas=datas,
    hiddenimports=fluent_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "matplotlib", "notebook", "pytest", "tkinter"],
    noarchive=False,
    optimize=1,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="ClassificadorIzhikevich",
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
    icon=str(icon_path),
    version=str(version_path),
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ClassificadorIzhikevich",
)
