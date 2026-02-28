"""Detect hardware characteristics for the current machine."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

from audio_transcribe.models import HardwareInfo


def detect_hardware() -> HardwareInfo:
    """Return hardware fingerprint for the current machine."""
    chip = _detect_chip()
    cores = os.cpu_count() or 1
    memory_gb = _detect_memory_gb()
    os_version = f"{platform.system()} {platform.release()}"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return HardwareInfo(
        chip=chip,
        cores_physical=cores,
        memory_gb=memory_gb,
        os=os_version,
        python=python_version,
    )


def _detect_chip() -> str:
    """Detect CPU/chip name. macOS: sysctl, Linux: /proc/cpuinfo, fallback: platform."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return platform.processor() or platform.machine()


def _detect_memory_gb() -> int:
    """Detect total RAM in GB. macOS: sysctl, Linux: /proc/meminfo, fallback: 0."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) // (1024**3)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024**2)
        except (FileNotFoundError, ValueError):
            pass
    return 0
