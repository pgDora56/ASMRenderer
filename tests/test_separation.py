"""音源分離まわりのテスト。

demucs 本体は重いのでテストでは実行せず、runner フックにフェイクを
差し替えてキャッシュ・エラーハンドリングのロジックを検証する。
"""

import numpy as np
import pytest
import soundfile as sf

from asmrenderer.separation import BGM_NAME, VOCALS_NAME, separate_vocals


@pytest.fixture
def input_wav(tmp_path, sr):
    path = tmp_path / "stream.wav"
    sf.write(path, np.zeros(sr, dtype=np.float32), sr)
    return path


def _fake_runner_factory(sr, calls):
    """呼び出し回数を記録しつつダミー stems を書き出すフェイク runner。"""

    def runner(input_path, out_dir, model):
        calls.append(model)
        sf.write(out_dir / VOCALS_NAME, np.zeros(sr, dtype=np.float32), sr)
        sf.write(out_dir / BGM_NAME, np.zeros((sr, 2), dtype=np.float32), sr)

    return runner


def test_separate_and_cache(input_wav, sr):
    """1 回目は分離が走り、2 回目はキャッシュが使われる。"""
    calls = []
    runner = _fake_runner_factory(sr, calls)

    vocals, bgm = separate_vocals(input_wav, runner=runner)
    assert vocals.exists() and bgm.exists()
    assert vocals.parent.name == "stream_stems"
    assert len(calls) == 1

    # 2 回目はキャッシュヒットで runner が呼ばれない
    separate_vocals(input_wav, runner=runner)
    assert len(calls) == 1


def test_force_reruns(input_wav, sr):
    calls = []
    runner = _fake_runner_factory(sr, calls)
    separate_vocals(input_wav, runner=runner)
    separate_vocals(input_wav, runner=runner, force=True)
    assert len(calls) == 2


def test_missing_output_raises(input_wav):
    """runner が stems を書かなかったらエラーになる。"""

    def broken_runner(input_path, out_dir, model):
        pass  # 何も書かない

    with pytest.raises(RuntimeError, match="分離結果"):
        separate_vocals(input_wav, runner=broken_runner)


def test_demucs_not_installed_message(input_wav, monkeypatch):
    """demucs 未インストール時にインストール方法を案内するエラーになる。"""
    import asmrenderer.separation as sep

    monkeypatch.setattr(sep.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="separate"):
        separate_vocals(input_wav, force=True)
