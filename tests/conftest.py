"""テスト共通のフィクスチャ。"""

import numpy as np
import pytest

from asmrenderer.vad import SpeechAnalysis

SR = 48000


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def speech_like_audio(sr) -> np.ndarray:
    """発話風の合成音声: ノイズバーストとポーズの繰り返し (60 秒)。

    [2s 発話, 1s ポーズ] × 20 のパターン。VAD・E2E テストで使う。
    """
    rng = np.random.default_rng(42)
    audio = np.zeros(60 * sr, dtype=np.float32)
    t_frame = np.arange(2 * sr) / sr
    for i in range(20):
        start = i * 3 * sr
        # 声っぽさのために 150Hz 基本波 + 倍音に AM をかけたもの + ノイズ
        burst = (
            0.3 * np.sin(2 * np.pi * 150 * t_frame)
            + 0.15 * np.sin(2 * np.pi * 450 * t_frame)
            + 0.05 * rng.standard_normal(2 * sr)
        )
        # 音節風の 4Hz AM
        burst *= 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t_frame) ** 2
        audio[start : start + 2 * sr] = burst.astype(np.float32)
    return audio * 0.5


@pytest.fixture
def long_analysis() -> SpeechAnalysis:
    """プランナーテスト用: 10 分の音源相当の発話区間 (12s 発話 / 1.5s ポーズ)。"""
    segments = []
    t = 0.0
    while t + 12.0 < 600.0:
        segments.append((t, t + 12.0))
        t += 13.5
    return SpeechAnalysis(duration=600.0, segments=segments)
