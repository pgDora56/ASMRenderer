"""YouTube 連携のテスト。yt-dlp / ffmpeg は呼ばず、フックで差し替える。"""

import json

import soundfile as sf
import pytest

from asmrenderer import cli
from asmrenderer.youtube import FetchedAudio, fetch_audio


def _fake_downloader(calls):
    def downloader(url, work_dir):
        calls.append(("download", url))
        src = work_dir / "abc123.webm"
        src.write_bytes(b"fake")
        return "abc123", "テスト動画", src

    return downloader


def _fake_converter(calls, audio, sr):
    def converter(src, dst, start, duration):
        calls.append(("convert", start, duration))
        sf.write(dst, audio, sr)

    return converter


def test_fetch_audio_names_and_caches(tmp_path, speech_like_audio, sr):
    """全編取得は <ID>.wav、範囲指定は <ID>_<start>s-<dur>s.wav。2 回目は変換しない。"""
    calls = []
    dl, cv = _fake_downloader(calls), _fake_converter(calls, speech_like_audio, sr)

    full = fetch_audio("https://youtu.be/abc123", tmp_path, downloader=dl, converter=cv)
    assert full.video_id == "abc123"
    assert full.wav_path == tmp_path / "abc123.wav"
    assert calls == [("download", "https://youtu.be/abc123"), ("convert", 0.0, None)]

    clip = fetch_audio(
        "https://youtu.be/abc123", tmp_path, start=600, duration=60, downloader=dl, converter=cv
    )
    assert clip.wav_path == tmp_path / "abc123_600s-60s.wav"
    assert calls[-1] == ("convert", 600, 60)

    # 同じ範囲をもう一度 → 変換は走らない
    n = len(calls)
    fetch_audio("https://youtu.be/abc123", tmp_path, start=600, duration=60, downloader=dl, converter=cv)
    assert [c for c in calls[n:] if c[0] == "convert"] == []


def test_yt_command_runs_pipeline(tmp_path, speech_like_audio, sr, monkeypatch):
    """yt コマンドは取得した wav に対して run と同じ出力を作る。"""
    wav = tmp_path / "abc123.wav"
    sf.write(wav, speech_like_audio, sr)

    seen = {}

    def fake_fetch(url, work_dir, start=0.0, duration=None):
        seen.update(url=url, work_dir=work_dir, start=start, duration=duration)
        return FetchedAudio("abc123", "テスト動画", tmp_path / "abc123.webm", wav)

    monkeypatch.setattr(cli, "fetch_audio", fake_fetch)
    ret = cli.main([
        "yt", "https://youtu.be/abc123", "--work-dir", str(tmp_path),
        "--start", "5", "--duration", "30", "--seed", "3",
    ])
    assert ret == 0
    assert seen == {"url": "https://youtu.be/abc123", "work_dir": str(tmp_path), "start": 5.0, "duration": 30.0}

    out = tmp_path / "abc123_asmr.wav"
    data, out_sr = sf.read(out)
    assert out_sr == sr and data.ndim == 2 and data.shape[1] == 2
    plan = json.loads((tmp_path / "abc123_asmr.plan.json").read_text())
    assert plan["params"]["seed"] == 3


def test_yt_command_reports_missing_tool(tmp_path, monkeypatch):
    """yt-dlp が無い場合はトレースバックではなくメッセージで終了する。"""
    def fake_fetch(url, work_dir, start=0.0, duration=None):
        raise RuntimeError("yt-dlp が見つかりません。")

    monkeypatch.setattr(cli, "fetch_audio", fake_fetch)
    with pytest.raises(SystemExit, match="yt-dlp"):
        cli.main(["yt", "https://youtu.be/abc123", "--work-dir", str(tmp_path)])
