"""Tests for the corrections stage — load, apply, and learn corrections."""

import yaml

from audio_transcribe.stages.correct import apply_corrections, learn_corrections, load_corrections


def test_load_corrections_empty(tmp_path):
    path = tmp_path / "corrections.yaml"
    c = load_corrections(str(path))
    assert c["substitutions"] == {}
    assert c["patterns"] == []


def test_load_corrections_from_file(tmp_path):
    path = tmp_path / "corrections.yaml"
    path.write_text(
        yaml.dump(
            {
                "substitutions": {"кубернетес": "Kubernetes"},
                "patterns": [],
            }
        )
    )
    c = load_corrections(str(path))
    assert c["substitutions"]["кубернетес"] == "Kubernetes"


def test_apply_corrections_substitution():
    corrections = {"substitutions": {"кубернетес": "Kubernetes"}, "patterns": []}
    segments = [{"text": "Мы используем кубернетес для деплоя", "start": 0.0, "end": 3.0}]
    result, count = apply_corrections(segments, corrections)
    assert "Kubernetes" in result[0]["text"]
    assert count == 1


def test_apply_corrections_case_insensitive():
    corrections = {"substitutions": {"кубернетес": "Kubernetes"}, "patterns": []}
    segments = [{"text": "Кубернетес работает", "start": 0.0, "end": 2.0}]
    result, count = apply_corrections(segments, corrections)
    assert "Kubernetes" in result[0]["text"]
    assert count == 1


def test_apply_corrections_word_level():
    corrections = {"substitutions": {"хелло": "hello"}, "patterns": []}
    segments = [
        {
            "text": "хелло world",
            "start": 0.0,
            "end": 1.0,
            "words": [
                {"word": "хелло", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.6, "end": 1.0},
            ],
        },
    ]
    result, count = apply_corrections(segments, corrections)
    assert result[0]["words"][0]["word"] == "hello"
    assert count == 1


def test_apply_corrections_pattern():
    corrections = {"substitutions": {}, "patterns": [{"match": "\\bпиар\\b", "replace": "PR"}]}
    segments = [{"text": "нужен пиар для проекта", "start": 0.0, "end": 2.0}]
    result, count = apply_corrections(segments, corrections)
    assert "PR" in result[0]["text"]
    assert count == 1


def test_apply_corrections_no_match():
    corrections = {"substitutions": {"nonexistent": "replacement"}, "patterns": []}
    segments = [{"text": "normal text", "start": 0.0, "end": 1.0}]
    result, count = apply_corrections(segments, corrections)
    assert result[0]["text"] == "normal text"
    assert count == 0


def test_learn_corrections_finds_diff():
    original = ["привет мир кубернетес"]
    corrected = ["привет мир Kubernetes"]
    learned = learn_corrections(original, corrected)
    assert "кубернетес" in learned
    assert learned["кубернетес"] == "Kubernetes"


def test_learn_corrections_multiple():
    original = ["кубернетес и дженкинс работают"]
    corrected = ["Kubernetes и Jenkins работают"]
    learned = learn_corrections(original, corrected)
    assert len(learned) == 2


def test_learn_corrections_no_diff():
    original = ["привет мир"]
    corrected = ["привет мир"]
    learned = learn_corrections(original, corrected)
    assert len(learned) == 0


def test_load_corrections_language_scoped(tmp_path):

    """Language-scoped corrections should load only the matching language."""
    corrections_file = tmp_path / "corrections.yaml"
    corrections_file.write_text(
        "ru:\n  substitutions:\n    кубернетес: Kubernetes\nen:\n  substitutions:\n    colour: color\n"
    )
    ru = load_corrections(str(corrections_file), language="ru")
    assert ru["substitutions"] == {"кубернетес": "Kubernetes"}
    en = load_corrections(str(corrections_file), language="en")
    assert en["substitutions"] == {"colour": "color"}


def test_load_corrections_legacy_flat_format(tmp_path):

    """Legacy flat format (no language keys) should still work."""
    corrections_file = tmp_path / "corrections.yaml"
    corrections_file.write_text("substitutions:\n  кубернетес: Kubernetes\n")
    result = load_corrections(str(corrections_file), language="ru")
    assert result["substitutions"] == {"кубернетес": "Kubernetes"}


def test_learn_corrections_unequal_word_count():
    """Replacements with different word counts should be captured as phrases."""
    original = ["в общем то да"]
    corrected = ["вообще да"]
    learned = learn_corrections(original, corrected)
    assert "в общем то" in learned
    assert learned["в общем то"] == "вообще"
