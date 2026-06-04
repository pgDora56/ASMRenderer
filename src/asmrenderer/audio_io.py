"""音声ファイルの入出力。

soundfile (libsndfile) で読める形式 (wav / flac / ogg / mp3 など) を直接読む。
読めない形式 (m4a 等) は、ffmpeg がインストールされていればそれを経由してデコードする。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """音声ファイルをチャンネル構成を保ったまま float32 で読み込む。

    Returns:
        (audio, sr): audio は shape (n, ch) の float32、sr はサンプルレート。
    """
    path = Path(path)
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        # libsndfile で読めない形式は ffmpeg にフォールバック
        data, sr = _load_via_ffmpeg(path)
    return data, sr


def load_audio_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """音声ファイルをモノラル float32 で読み込む。

    Returns:
        (audio, sr): audio は shape (n,) の float32、sr はサンプルレート。
    """
    data, sr = load_audio(path)
    # ステレオ以上はミックスダウンしてモノラル化
    mono = data.mean(axis=1).astype(np.float32)
    return mono, sr


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """ffmpeg で一時 wav にデコードしてから読み込む。"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path.suffix} 形式の読み込みには ffmpeg が必要です。"
            "`brew install ffmpeg` でインストールするか、wav/flac/mp3 に変換してください。"
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "decoded.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", str(tmp_wav)],
            check=True,
            capture_output=True,
        )
        data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
    return data, sr


def save_audio(path: str | Path, data: np.ndarray, sr: int) -> None:
    """ステレオ音声を書き出す。拡張子に応じて wav (24bit) / flac を選ぶ。"""
    path = Path(path)
    subtype = "PCM_24" if path.suffix.lower() == ".wav" else None
    sf.write(path, data, sr, subtype=subtype)
