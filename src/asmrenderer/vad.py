"""発話区間検出 (VAD: Voice Activity Detection)。

移動プランナーが「会話の切れ目」を知るために使う。手法は 2 つ:

- energy: RMS エネルギーベース。依存なしで動くが、BGM が大きい配信では精度が落ちる。
- silero: Silero VAD (pysilero-vad, optional extra)。BGM 入りでも声だけを拾える。
  `pip install 'asmrenderer[silero]'` で有効化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage, signal


@dataclass
class SpeechAnalysis:
    """発話区間の解析結果。時刻はすべて秒。"""

    duration: float
    segments: list[tuple[float, float]] = field(default_factory=list)
    method: str = "energy"

    def pauses(self, min_len: float = 0.0) -> list[tuple[float, float]]:
        """発話区間の「間」(ポーズ) を返す。冒頭・末尾の無音は含めない。

        移動イベントのトリガー候補になるので、min_len で短すぎる間を除外できる。
        """
        result = []
        for (_, prev_end), (next_start, _) in zip(self.segments, self.segments[1:]):
            if next_start - prev_end >= min_len:
                result.append((prev_end, next_start))
        return result

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "duration": self.duration,
                "method": self.method,
                "segments": [[round(s, 3), round(e, 3)] for s, e in self.segments],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "SpeechAnalysis":
        obj = json.loads(text)
        return cls(
            duration=obj["duration"],
            segments=[(s, e) for s, e in obj["segments"]],
            method=obj.get("method", "unknown"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "SpeechAnalysis":
        return cls.from_json(Path(path).read_text())


def detect_speech(
    audio: np.ndarray,
    sr: int,
    method: str = "energy",
    min_speech_s: float = 0.3,
    min_gap_s: float = 0.25,
) -> SpeechAnalysis:
    """発話区間を検出する。

    Args:
        min_speech_s: これより短い発話区間は捨てる
        min_gap_s: これより短いギャップは前後の発話区間と結合する
    """
    if method == "energy":
        speech_mask, hop_s = _detect_energy(audio, sr)
    elif method == "silero":
        speech_mask, hop_s = _detect_silero(audio, sr)
    else:
        raise ValueError(f"未対応の VAD 手法: {method}")

    segments = _mask_to_segments(speech_mask, hop_s, min_speech_s, min_gap_s)
    return SpeechAnalysis(duration=len(audio) / sr, segments=segments, method=method)


def _detect_energy(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """RMS エネルギーベースの簡易 VAD。

    閾値は「発話レベル (上位 95 パーセンタイル) から -20dB」を基準にする。
    雑談配信はほぼ喋りっぱなしで無音が 1 割を切ることも多く、
    下位パーセンタイルによるノイズフロア推定は破綻するため、上から取る。
    """
    frame_s, hop_s = 0.03, 0.01
    frame = int(frame_s * sr)
    hop = int(hop_s * sr)

    n_frames = max(1, (len(audio) - frame) // hop + 1)
    # フレームごとの RMS をベクトル演算で計算 (sliding_window_view はメモリビューなので軽い)
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop][:n_frames]
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
    rms_db = 20 * np.log10(rms + 1e-10)

    # 発話レベル基準の閾値。-60dBFS の絶対フロアより下には下げない
    # (全編無音・微小ノイズだけの入力で全フレーム発話扱いになるのを防ぐ)
    speech_level = np.percentile(rms_db, 95)
    threshold = max(speech_level - 20.0, -60.0)
    mask = rms_db > threshold

    # ハングオーバー: 発話の前後 150ms を発話側に倒して、語尾の途切れを防ぐ
    hangover = int(0.15 / hop_s)
    mask = ndimage.binary_dilation(mask, iterations=hangover)
    return mask, hop_s


def _detect_silero(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Silero VAD (機械学習ベース)。BGM 入りの配信音源向け。"""
    try:
        from pysilero_vad import SileroVoiceActivityDetector
    except ImportError as e:
        raise RuntimeError(
            "silero VAD を使うには `pip install 'asmrenderer[silero]'` が必要です"
        ) from e

    # Silero は 16kHz / 512 サンプル単位 (32ms) で処理する
    target_sr = 16000
    if sr != target_sr:
        audio16 = signal.resample_poly(audio, target_sr, sr)
    else:
        audio16 = audio
    pcm16 = np.clip(audio16 * 32767, -32768, 32767).astype(np.int16)

    vad = SileroVoiceActivityDetector()
    chunk = 512
    n_chunks = len(pcm16) // chunk
    probs = np.empty(n_chunks)
    for i in range(n_chunks):
        probs[i] = vad(pcm16[i * chunk : (i + 1) * chunk].tobytes())

    hop_s = chunk / target_sr
    mask = probs > 0.5
    # energy 側と同様のハングオーバー
    mask = ndimage.binary_dilation(mask, iterations=max(1, int(0.15 / hop_s)))
    return mask, hop_s


def _mask_to_segments(
    mask: np.ndarray, hop_s: float, min_speech_s: float, min_gap_s: float
) -> list[tuple[float, float]]:
    """フレーム単位の発話マスクを (start, end) 秒のリストに変換する。"""
    if not mask.any():
        return []

    # マスクの立ち上がり / 立ち下がりを検出
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0] * hop_s
    ends = np.where(diff == -1)[0] * hop_s

    # 短いギャップは結合
    merged: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # 短すぎる発話区間は捨てる
    return [(s, e) for s, e in merged if e - s >= min_speech_s]
