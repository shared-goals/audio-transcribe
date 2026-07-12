import pytest

from audio_transcribe.config import Backend, Profile, load_profile


def test_default_profile():
    profile = load_profile()
    assert profile.backend == Backend.MLX_VAD
    assert profile.no_diarize is True


def test_load_named_profile(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[profiles.fast]\nlanguage = "en"\nmodel = "small"\nbackend = "mlx"\n')
    profile = load_profile("fast", config)
    assert profile.language == "en"
    assert profile.model == "small"
    assert profile.backend == "mlx"


@pytest.mark.parametrize(
    "profile, message",
    [
        (Profile(backend="typo"), "unsupported backend"),
        (Profile(min_speakers=0), "at least 1"),
        (Profile(min_speakers=4, max_speakers=2), "greater than or equal"),
        (Profile(language=""), "must not be empty"),
    ],
)
def test_profile_validation(profile, message):
    with pytest.raises(ValueError, match=message):
        profile.validated()


def test_rejects_unknown_profile_key(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[profiles.default]\ntyop = true\n")
    with pytest.raises(ValueError, match="unknown profile option"):
        load_profile(path=config)
