"""Tests for hardware detection."""

from audio_transcribe.models import HardwareInfo
from audio_transcribe.stats.hardware import detect_hardware


def test_detect_hardware_returns_hardware_info():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.cores_physical > 0
    assert hw.memory_gb > 0
    assert len(hw.chip) > 0
    assert len(hw.os) > 0
    assert len(hw.python) > 0
