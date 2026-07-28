"""Build and repair helpers for a coherent FFmpeg 7 runtime on macOS."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path


def ffmpeg7_build_environment(prefix: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment that builds PyAV from source against FFmpeg 7."""
    env = dict(base if base is not None else os.environ)
    env.update(
        {
            "UV_NO_BINARY_PACKAGE": "av",
            "PKG_CONFIG_PATH": str(prefix / "lib" / "pkgconfig"),
            "CPPFLAGS": f"-I{prefix / 'include'}",
            "LDFLAGS": f"-L{prefix / 'lib'}",
        }
    )
    return env


def patch_torchcodec_rpath(package_dir: Path, prefix: Path) -> int:
    """Add the FFmpeg 7 RPATH to TorchCodec binaries and ad-hoc sign them."""
    targets = sorted((*package_dir.glob("*7.dylib"), *package_dir.glob("*7.so")))
    rpath = str(prefix / "lib")
    for target in targets:
        subprocess.run(
            ["install_name_tool", "-add_rpath", rpath, str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["codesign", "--force", "--sign", "-", str(target)],
            capture_output=True,
            text=True,
            check=True,
        )
    return len(targets)


def main() -> None:
    """Repair TorchCodec in the active Python environment."""
    if sys.platform != "darwin":
        return
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m audio_transcribe.macos_ffmpeg /path/to/ffmpeg@7")
    package_dir = Path(sysconfig.get_paths()["purelib"]) / "torchcodec"
    patched = patch_torchcodec_rpath(package_dir, Path(sys.argv[1]))
    if patched == 0:
        raise SystemExit(f"no FFmpeg 7 TorchCodec libraries found under {package_dir}")


if __name__ == "__main__":
    main()
