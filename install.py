import launch
from importlib import metadata


rembg_expected_version = "2.0.59"


try:
    rembg_installed_version = metadata.version("rembg")
except Exception:
    rembg_installed_version = None


if rembg_installed_version != rembg_expected_version:
    launch.run_pip(f"install rembg=={rembg_expected_version} --no-deps", "rembg")


for dep in ['onnxruntime', 'pymatting', 'pooch']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for REMBG extension")
