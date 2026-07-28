"""Tests for the coherent macOS FFmpeg 7 runtime."""

from pathlib import Path
from unittest.mock import call, patch

from audio_transcribe.macos_ffmpeg import ffmpeg7_build_environment, patch_torchcodec_rpath


def test_ffmpeg7_build_environment() -> None:
    env = ffmpeg7_build_environment(Path("/opt/homebrew/opt/ffmpeg@7"), {"PATH": "/bin"})

    assert env["UV_NO_BINARY_PACKAGE"] == "av"
    assert env["PKG_CONFIG_PATH"] == "/opt/homebrew/opt/ffmpeg@7/lib/pkgconfig"
    assert env["CPPFLAGS"] == "-I/opt/homebrew/opt/ffmpeg@7/include"
    assert env["LDFLAGS"] == "-L/opt/homebrew/opt/ffmpeg@7/lib"
    assert env["PATH"] == "/bin"


def test_patch_torchcodec_rpath_patches_and_resigns_ffmpeg7_libraries(tmp_path: Path) -> None:
    package = tmp_path / "torchcodec"
    package.mkdir()
    core = package / "libtorchcodec_core7.dylib"
    pybind = package / "libtorchcodec_pybind_ops7.so"
    ignored = package / "libtorchcodec_core6.dylib"
    for artifact in (core, pybind, ignored):
        artifact.touch()

    with patch("audio_transcribe.macos_ffmpeg.subprocess.run") as run:
        patched = patch_torchcodec_rpath(package, Path("/opt/homebrew/opt/ffmpeg@7"))

    assert patched == 2
    assert run.call_args_list == [
        call(
            ["install_name_tool", "-add_rpath", "/opt/homebrew/opt/ffmpeg@7/lib", str(core)],
            capture_output=True,
            text=True,
            check=False,
        ),
        call(["codesign", "--force", "--sign", "-", str(core)], capture_output=True, text=True, check=True),
        call(
            ["install_name_tool", "-add_rpath", "/opt/homebrew/opt/ffmpeg@7/lib", str(pybind)],
            capture_output=True,
            text=True,
            check=False,
        ),
        call(["codesign", "--force", "--sign", "-", str(pybind)], capture_output=True, text=True, check=True),
    ]
