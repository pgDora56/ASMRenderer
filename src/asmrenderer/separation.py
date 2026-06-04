"""音源分離: BGM 入りの配信音声から「声」と「それ以外 (BGM)」を分ける。

Demucs (htdemucs) を使う。optional extra:
    uv pip install -e '.[separate]'

分離は重い処理 (CPU で実時間の数十%〜、CUDA GPU なら数分) なので、
結果は入力ファイルの隣の `<入力名>_stems/` にキャッシュして使い回す。

注意: Demucs の出力サンプルレートはモデル準拠 (htdemucs は 44.1kHz)。
入力が 48kHz でも stems は 44.1kHz になるが、後段はどのレートでも動く。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

# 分離結果のファイル名 (demucs --two-stems=vocals の出力名に合わせる)
VOCALS_NAME = "vocals.wav"
BGM_NAME = "no_vocals.wav"


def separate_vocals(
    input_path: str | Path,
    model: str = "htdemucs",
    force: bool = False,
    runner: Callable[[Path, Path, str], None] | None = None,
) -> tuple[Path, Path]:
    """声と BGM を分離し、(vocals_path, bgm_path) を返す。

    Args:
        force: True ならキャッシュを無視して分離し直す
        runner: 分離処理の実体 (テストで差し替えるためのフック)。
            (input_path, out_dir, model) を受け取り out_dir に
            vocals.wav / no_vocals.wav を書き出す契約。
    """
    input_path = Path(input_path)
    runner = runner or _run_demucs

    stems_dir = input_path.parent / f"{input_path.stem}_stems"
    vocals = stems_dir / VOCALS_NAME
    bgm = stems_dir / BGM_NAME

    # キャッシュヒット: 既に両 stems があれば再分離しない
    if not force and vocals.exists() and bgm.exists():
        return vocals, bgm

    stems_dir.mkdir(exist_ok=True)
    runner(input_path, stems_dir, model)

    if not (vocals.exists() and bgm.exists()):
        raise RuntimeError(f"分離結果が見つかりません: {stems_dir}")
    return vocals, bgm


def _run_demucs(input_path: Path, out_dir: Path, model: str) -> None:
    """demucs CLI を呼び出して 2-stem 分離を実行する。

    Python API ではなく CLI 経由にしているのは、demucs のバージョン間で
    API が変わっても影響を受けにくくするため。
    """
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError(
            "音源分離には demucs が必要です。"
            "`uv pip install -e '.[separate]'` でインストールしてください。"
            "(Windows + NVIDIA GPU の場合は CUDA 版 PyTorch を先に入れると高速です。"
            "詳細は README を参照)"
        )

    # demucs は <tmp>/<model>/<トラック名>/{vocals,no_vocals}.wav に書き出す
    tmp_root = out_dir / "_demucs_tmp"
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "-o",
        str(tmp_root),
        str(input_path),
    ]
    print(f"音源分離 (demucs/{model}) を実行中... 初回はモデルのダウンロードが入ります", file=sys.stderr)
    subprocess.run(cmd, check=True)

    track_dir = tmp_root / model / input_path.stem
    for name in (VOCALS_NAME, BGM_NAME):
        src = track_dir / name
        if not src.exists():
            raise RuntimeError(f"demucs の出力が想定と異なります: {src} がありません")
        shutil.move(str(src), out_dir / name)
    shutil.rmtree(tmp_root)
