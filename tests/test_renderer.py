"""バイノーラルレンダラーのテスト。"""

import numpy as np
import pytest
from scipy import signal

from asmrenderer.planner import MoveEvent, Plan, PlannerParams
from asmrenderer.renderer import RendererParams, render
from asmrenderer.trajectory import Trajectory


def _static_plan(az, dist=1.0, duration=3.0):
    """指定位置に静止し続けるプラン (揺らぎなし)。"""
    params = PlannerParams(seed=0, sway_az_deg=0.0, sway_dist_m=0.0)
    return Plan(duration=duration, init_az=az, init_dist=dist, params=params, events=[])


@pytest.fixture
def noise(sr):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(3 * sr) * 0.2).astype(np.float32)


def test_output_shape_and_validity(noise, sr):
    out = render(noise, sr, Trajectory(_static_plan(az=0.0)))
    assert out.shape == (len(noise), 2)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0  # クリップしていない


def test_right_source_is_louder_on_right(noise, sr):
    """音源が右 (+90°) にあるとき、右チャンネルの方が大きい。"""
    out = render(noise, sr, Trajectory(_static_plan(az=90.0)))
    mid = slice(sr, 2 * sr)  # 端の影響を避けて中央 1 秒で評価
    rms_l = np.sqrt(np.mean(out[mid, 0] ** 2))
    rms_r = np.sqrt(np.mean(out[mid, 1] ** 2))
    assert rms_r > rms_l * 1.3


def test_itd_direction_and_magnitude(noise, sr):
    """音源が右にあるとき、左チャンネルが ITD 分だけ遅れる。

    dist=1m, az=90° では到達距離差 ≈ 0.175m → 約 0.51ms ≈ 24.5 サンプル @48kHz。
    """
    out = render(noise, sr, Trajectory(_static_plan(az=90.0, dist=1.0)))
    mid = slice(sr, 2 * sr)
    corr = signal.correlate(out[mid, 0], out[mid, 1], mode="full")
    lag = np.argmax(corr) - (sr - 1)  # 正なら L が R より遅れている
    assert 18 <= lag <= 30


def test_front_is_symmetric(noise, sr):
    """正面 (0°) では左右がほぼ同等。"""
    out = render(noise, sr, Trajectory(_static_plan(az=0.0)))
    mid = slice(sr, 2 * sr)
    rms_l = np.sqrt(np.mean(out[mid, 0] ** 2))
    rms_r = np.sqrt(np.mean(out[mid, 1] ** 2))
    assert abs(rms_l - rms_r) / rms_l < 0.05


def test_near_is_louder_than_far(noise, sr):
    """近い音源の方が (正規化前の比率として) ドライ成分が大きい = 左右差より距離感。

    正規化があるので絶対値は比べられない。代わりに wet/dry 比で距離感を確認する:
    遠い方が相対的に残響が多く、自己相関の裾が広がる。簡易にスペクトル重心ではなく
    直接、近距離・遠距離のレンダリング結果の上位ピークの集中度で比較する。
    """
    near = render(noise, sr, Trajectory(_static_plan(az=0.0, dist=0.5)))
    far = render(noise, sr, Trajectory(_static_plan(az=0.0, dist=2.4)))
    # 残響が多いほど信号のエネルギーが時間方向に拡散する → 振幅分布の尖度が下がる
    mid = slice(sr, 2 * sr)

    def crest(x):
        return np.max(np.abs(x)) / np.sqrt(np.mean(x**2))

    # 同一ノイズ由来なので、遠い方が crest factor が下がる傾向を確認 (緩い閾値)
    assert crest(far[mid, 0]) < crest(near[mid, 0]) * 1.1


def test_face_away_muffles(noise, sr):
    """よそ見 (face=1) では高域が減ってスペクトル重心が下がる。"""
    plan_front = _static_plan(az=0.0)
    plan_away = _static_plan(az=0.0)
    plan_away.events = [
        MoveEvent("face_away", 0.0, 0.1, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    ]

    out_front = render(noise, sr, Trajectory(plan_front))
    out_away = render(noise, sr, Trajectory(plan_away))
    mid = slice(sr, 2 * sr)

    def centroid(x):
        f, p = signal.welch(x, sr, nperseg=2048)
        return np.sum(f * p) / np.sum(p)

    assert centroid(out_away[mid, 0]) < centroid(out_front[mid, 0]) * 0.8


def test_moving_source_no_artifacts(sr):
    """移動中 (左→右スイープ) でも出力が有限でクリップしない。"""
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(5 * sr) * 0.2).astype(np.float32)
    params = PlannerParams(seed=0, sway_az_deg=0.0, sway_dist_m=0.0)
    plan = Plan(
        duration=5.0,
        init_az=-90.0,
        init_dist=1.0,
        params=params,
        events=[MoveEvent("walk_talk", 1.0, 4.0, -90.0, 90.0, 1.0, 0.6, 0.0, 0.0)],
    )
    out = render(audio, sr, Trajectory(plan))
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0
    # 前半は左が大きく、後半は右が大きい
    head = slice(0, sr)
    tail = slice(4 * sr, 5 * sr)
    assert np.sqrt(np.mean(out[head, 0] ** 2)) > np.sqrt(np.mean(out[head, 1] ** 2))
    assert np.sqrt(np.mean(out[tail, 1] ** 2)) > np.sqrt(np.mean(out[tail, 0] ** 2))


def test_t_offset_matches_full_render(noise, sr):
    """部分レンダリング (t_offset) が全体レンダリングの対応区間と一致する方向性を持つ。

    正規化が独立に走るので波形一致までは求めず、左右バランスが一致することを見る。
    """
    params = PlannerParams(seed=0, sway_az_deg=0.0, sway_dist_m=0.0)
    plan = Plan(
        duration=3.0,
        init_az=-60.0,
        init_dist=1.0,
        params=params,
        events=[MoveEvent("pause_jump", 1.0, 1.5, -60.0, 60.0, 1.0, 1.0, 0.0, 0.0)],
    )
    traj = Trajectory(plan)
    full = render(noise, sr, traj)
    partial = render(noise[2 * sr :], sr, traj, t_offset=2.0)

    def balance(x):  # R/L 比
        return np.sqrt(np.mean(x[:, 1] ** 2)) / np.sqrt(np.mean(x[:, 0] ** 2))

    assert balance(partial) == pytest.approx(balance(full[2 * sr :]), rel=0.05)
    # 2 秒以降は az=+60 に居るはずなので右が大きい
    assert balance(partial) > 1.2
