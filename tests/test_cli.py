"""CLI の E2E テスト。"""

import json

import numpy as np
import soundfile as sf
import pytest

from asmrenderer.cli import main


@pytest.fixture
def input_wav(tmp_path, speech_like_audio, sr):
    path = tmp_path / "input.wav"
    sf.write(path, speech_like_audio, sr)
    return path


def test_run_end_to_end(tmp_path, input_wav, sr):
    """run コマンドで入力からステレオ出力とプラン JSON が生成される。"""
    out = tmp_path / "out.wav"
    ret = main(["run", str(input_wav), "-o", str(out), "--seed", "1"])
    assert ret == 0

    data, out_sr = sf.read(out)
    assert out_sr == sr
    assert data.ndim == 2 and data.shape[1] == 2
    assert data.shape[0] == 60 * sr

    # プラン JSON も保存されている
    plan_path = tmp_path / "out.plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text())
    assert plan["duration"] == pytest.approx(60.0, abs=0.1)


def test_run_preview(tmp_path, input_wav, sr):
    """--start/--duration で部分レンダリングできる。"""
    out = tmp_path / "preview.wav"
    ret = main(["run", str(input_wav), "-o", str(out), "--start", "10", "--duration", "5"])
    assert ret == 0
    data, _ = sf.read(out)
    assert data.shape[0] == 5 * sr


def test_analyze_plan_render_pipeline(tmp_path, input_wav, sr):
    """analyze → plan → render を個別に繋いでも動く。"""
    seg = tmp_path / "segments.json"
    plan = tmp_path / "plan.json"
    out = tmp_path / "out.flac"

    assert main(["analyze", str(input_wav), "-o", str(seg)]) == 0
    assert seg.exists()

    assert main(["plan", str(seg), "-o", str(plan), "--seed", "2"]) == 0
    assert plan.exists()

    assert main(["render", str(input_wav), str(plan), "-o", str(out)]) == 0
    data, _ = sf.read(out)
    assert data.shape == (60 * sr, 2)


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_run_with_separation(tmp_path, speech_like_audio, sr, monkeypatch):
    """--separate で声だけ移動 + BGM 静止ミックスのパイプラインが通る。"""
    # BGM 入りの入力 (声 + サイン波 BGM)
    rng = np.random.default_rng(0)
    t = np.arange(len(speech_like_audio)) / sr
    bgm_src = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    in_path = tmp_path / "stream.wav"
    sf.write(in_path, speech_like_audio + bgm_src, sr)

    # 分離はフェイク: 声 stem = フィクスチャ音声、BGM stem = サイン波ステレオ
    def fake_separate(input_path, force=False, **kw):
        stems = tmp_path / "stream_stems"
        stems.mkdir(exist_ok=True)
        v, b = stems / "vocals.wav", stems / "no_vocals.wav"
        sf.write(v, speech_like_audio, sr)
        sf.write(b, np.stack([bgm_src, bgm_src], axis=1), sr)
        return v, b

    import asmrenderer.cli as cli_mod

    monkeypatch.setattr(cli_mod, "separate_vocals", fake_separate)

    out = tmp_path / "out.wav"
    # seed=2 は 60 秒のフィクスチャでも移動イベントが 1 件以上出る (プランナーは決定論的)
    ret = main(["run", str(in_path), "-o", str(out), "--separate", "--bgm-gain-db", "-6",
                "--seed", "2"])
    assert ret == 0

    data, out_sr = sf.read(out)
    assert out_sr == sr
    assert data.shape == (60 * sr, 2)
    assert np.max(np.abs(data)) <= 1.0

    # プランは声 stem から作られている (= BGM に邪魔されず発話区間が取れている)
    plan = json.loads((tmp_path / "out.plan.json").read_text())
    assert len(plan["events"]) >= 1


def test_separate_command(tmp_path, sr, monkeypatch):
    """separate サブコマンド単体でも stems が作れる。"""
    in_path = tmp_path / "stream.wav"
    sf.write(in_path, np.zeros(sr, dtype=np.float32), sr)

    import asmrenderer.separation as sep

    def fake_runner(input_path, out_dir, model):
        sf.write(out_dir / "vocals.wav", np.zeros(sr, dtype=np.float32), sr)
        sf.write(out_dir / "no_vocals.wav", np.zeros((sr, 2), dtype=np.float32), sr)

    monkeypatch.setattr(sep, "_run_demucs", fake_runner)
    ret = main(["separate", str(in_path)])
    assert ret == 0
    assert (tmp_path / "stream_stems" / "vocals.wav").exists()
