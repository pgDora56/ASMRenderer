"""VAD (発話区間検出) のテスト。"""

import numpy as np
import pytest

from asmrenderer.vad import SpeechAnalysis, detect_speech


def test_energy_vad_detects_bursts(speech_like_audio, sr):
    """合成音声の発話バーストがほぼ正しい位置で検出される。"""
    analysis = detect_speech(speech_like_audio, sr, method="energy")

    assert analysis.duration == pytest.approx(60.0, abs=0.1)
    # 20 個のバーストが (多少の結合は許容しつつ) ほぼ検出される
    assert 15 <= len(analysis.segments) <= 20

    # 最初のバースト (0-2s) がおおよそ捉えられている
    s0, e0 = analysis.segments[0]
    assert s0 < 0.5
    assert 1.5 < e0 < 2.7


def test_energy_vad_silence():
    """無音のみの入力では発話区間が出ない。"""
    sr = 16000
    silence = np.zeros(5 * sr, dtype=np.float32)
    analysis = detect_speech(silence, sr, method="energy")
    assert analysis.segments == []


def test_pauses(speech_like_audio, sr):
    """ポーズ (発話間のギャップ) が抽出できる。"""
    analysis = detect_speech(speech_like_audio, sr, method="energy")
    pauses = analysis.pauses(min_len=0.5)
    assert len(pauses) >= 10
    for ps, pe in pauses:
        assert pe - ps >= 0.5


def test_json_roundtrip():
    """解析結果の JSON 保存・復元。"""
    analysis = SpeechAnalysis(duration=100.0, segments=[(0.5, 8.2), (9.0, 20.0)])
    restored = SpeechAnalysis.from_json(analysis.to_json())
    assert restored.duration == 100.0
    assert restored.segments == [(0.5, 8.2), (9.0, 20.0)]


def test_unknown_method(sr):
    with pytest.raises(ValueError):
        detect_speech(np.zeros(sr), sr, method="nonexistent")
