"""YouTube 連携: URL から音声を取得して変換パイプラインに渡す。

yt-dlp と ffmpeg の CLI をサブプロセスで呼ぶ (どちらも PATH にある前提)。
取得した音声は作業ディレクトリに <動画ID>.<ext> として置き、ffmpeg で
ステレオ 48kHz の wav に変換してから `run` と同じ処理に流す。

長尺配信 (2〜3 時間) は wav にすると 1GB 超になり、パイプラインは全編を
メモリに読むため、`--start` / `--duration` で ffmpeg の段階で切り出せる。
切り出し済み wav と元のダウンロードは作業ディレクトリにキャッシュされ、
同じ URL・同じ範囲なら 2 回目以降は再取得しない。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# フック型: テストで実体を差し替えるため
Downloader = Callable[[str, Path], tuple[str, str, Path]]
Converter = Callable[[Path, Path, float, float | None], None]


@dataclass
class FetchedAudio:
    video_id: str
    title: str
    source_path: Path  # yt-dlp が落とした元ファイル (webm/m4a など)
    wav_path: Path  # 変換 (+ 切り出し) 済み wav


def fetch_audio(
    url: str,
    work_dir: str | Path,
    start: float = 0.0,
    duration: float | None = None,
    downloader: Downloader | None = None,
    converter: Converter | None = None,
) -> FetchedAudio:
    """URL の音声を取得し、変換パイプラインに渡せる wav のパスを返す。"""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    downloader = downloader or _run_yt_dlp
    converter = converter or _run_ffmpeg

    video_id, title, source_path = downloader(url, work_dir)

    suffix = ""
    if start > 0 or duration is not None:
        suffix = f"_{int(start)}s" + (f"-{int(duration)}s" if duration is not None else "")
    wav_path = work_dir / f"{video_id}{suffix}.wav"
    if not wav_path.exists():
        converter(source_path, wav_path, start, duration)
    else:
        print(f"変換済み wav を使用: {wav_path}", file=sys.stderr)
    return FetchedAudio(video_id, title, source_path, wav_path)


def _require(cmd: str, hint: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"{cmd} が見つかりません。{hint}")


def _run_yt_dlp(url: str, work_dir: Path) -> tuple[str, str, Path]:
    """yt-dlp で音声だけを取得する。既に同じ ID のファイルがあれば再利用する。"""
    _require("yt-dlp", "`pip install yt-dlp` または `uv tool install yt-dlp` でインストールしてください。")
    # まずメタデータだけ取得 (ID とタイトル)。--print は暗黙に --simulate になる
    meta = _run_captured(
        ["yt-dlp", "--no-update", "--no-playlist", "--print", "id", "--print", "title", url]
    ).splitlines()
    if len(meta) < 2:
        raise RuntimeError(f"yt-dlp から動画情報を取得できませんでした: {url}")
    video_id, title = meta[0].strip(), meta[1].strip()

    existing = [p for p in work_dir.glob(f"{video_id}.*") if p.suffix != ".wav"]
    if existing:
        print(f"ダウンロード済み: {existing[0]}", file=sys.stderr)
        return video_id, title, existing[0]

    print(f"ダウンロード: {title} ({video_id})", file=sys.stderr)
    # --print を使うと --simulate が暗黙に付くので --no-simulate で実際に落とす
    out = _run_captured(
        [
            "yt-dlp", "--no-update", "--no-playlist", "-f", "bestaudio", "--no-progress",
            "--no-simulate", "--print", "after_move:filepath",
            "-o", str(work_dir / f"{video_id}.%(ext)s"),
            url,
        ]
    ).strip().splitlines()
    if not out:
        raise RuntimeError("yt-dlp の出力先パスを取得できませんでした")
    return video_id, title, Path(out[-1])


def _run_captured(cmd: list[str]) -> str:
    """stdout を返す。失敗時は stderr の末尾を含めた RuntimeError にする。"""
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = chr(10).join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"{cmd[0]} が失敗しました (exit {proc.returncode}):" + chr(10) + tail)
    return proc.stdout


def _run_ffmpeg(src: Path, dst: Path, start: float, duration: float | None) -> None:
    """ffmpeg でステレオ 48kHz wav に変換する (必要なら切り出し)。"""
    _require("ffmpeg", "https://ffmpeg.org/ からインストールして PATH に通してください。")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(src), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(dst)]
    rng = f" ({start:.0f}s から{f' {duration:.0f}s' if duration is not None else ' 最後まで'})" if (start > 0 or duration is not None) else ""
    print(f"wav に変換{rng}: {dst}", file=sys.stderr)
    subprocess.run(cmd, check=True)
