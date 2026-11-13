from audio_transcribe.stages.cluster_hygiene import (
    diarization_report,
    merge_small_clusters,
    small_clusters,
)


def seg(start, end, speaker):
    return {"start": start, "end": end, "text": "x", "speaker": speaker}


def test_an_eight_second_artefact_is_not_a_speaker() -> None:
    segments = [
        seg(0, 300, "SPEAKER_00"),
        seg(300, 302, "SPEAKER_02"),
        seg(302, 600, "SPEAKER_01"),
    ]
    assert small_clusters(segments) == ["SPEAKER_02"]

    merged, changes = merge_small_clusters(segments)
    # 301 s is marginally closer to SPEAKER_01 (mid 451) than to SPEAKER_00 (mid 150).
    assert [s["speaker"] for s in merged] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    assert changes == [{"from": "SPEAKER_02", "to": "SPEAKER_01", "segments": 1, "seconds": 2.0}]


def test_unknown_is_never_treated_as_a_small_cluster() -> None:
    segments = [seg(0, 60, "SPEAKER_00"), seg(60, 61, "UNKNOWN")]
    assert small_clusters(segments) == []
    merged, changes = merge_small_clusters(segments)
    assert merged == segments and changes == []


def test_report_warns_about_short_clusters_and_unknown_share() -> None:
    segments = [
        seg(0, 100, "SPEAKER_00"),
        seg(100, 110, "SPEAKER_01"),
        seg(110, 130, "UNKNOWN"),
    ]
    report = diarization_report(segments, expected_speakers=3)
    assert report["unknown_share"] == 0.154
    assert any("UNKNOWN 15%" in warning for warning in report["warnings"])
    assert any("SPEAKER_01" in warning for warning in report["warnings"])
    assert any("ожидалось 3" in warning for warning in report["warnings"])
    assert report["clusters"][0]["speaker"] == "SPEAKER_00"


def test_report_is_quiet_for_a_clean_two_speaker_call() -> None:
    segments = [seg(0, 200, "SPEAKER_00"), seg(200, 400, "SPEAKER_01"), seg(400, 401, "UNKNOWN")]
    report = diarization_report(segments, expected_speakers=2)
    assert report["warnings"] == []
    assert report["unknown_share"] < 0.01
