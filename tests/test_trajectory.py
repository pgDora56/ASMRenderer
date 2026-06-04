"""トラジェクトリ (位置カーブ) のテスト。"""

import numpy as np

from asmrenderer.planner import MoveEvent, Plan, PlannerParams
from asmrenderer.trajectory import Trajectory


def _plan_without_sway(events, duration=60.0, init_az=0.0, init_dist=1.0):
    """揺らぎを切ったプラン (補間の検証がしやすい)。"""
    params = PlannerParams(seed=0, sway_az_deg=0.0, sway_dist_m=0.0)
    return Plan(
        duration=duration, init_az=init_az, init_dist=init_dist, params=params, events=events
    )


def test_hold_before_and_after_event():
    """イベント前は初期値、イベント後は終了値でホールドされる。"""
    ev = MoveEvent("pause_jump", 10.0, 13.0, 0.0, 90.0, 1.0, 0.5, 0.0, 0.0)
    traj = Trajectory(_plan_without_sway([ev]))

    az, dist, face = traj.sample(np.array([0.0, 5.0, 9.9, 20.0, 59.0]))
    np.testing.assert_allclose(az[:3], 0.0, atol=0.5)  # イベント前
    np.testing.assert_allclose(az[3:], 90.0, atol=0.5)  # イベント後
    np.testing.assert_allclose(dist[3:], 0.5, atol=0.01)
    np.testing.assert_allclose(face, 0.0)


def test_transition_is_monotonic_and_smooth():
    """遷移区間内で値が単調に変化し、端点で目標値に達する。"""
    ev = MoveEvent("walk_talk", 10.0, 20.0, 0.0, 100.0, 1.0, 1.0, 0.0, 0.0)
    traj = Trajectory(_plan_without_sway([ev]))

    t = np.linspace(10.0, 20.0, 200)
    az, _, _ = traj.sample(t)
    assert np.all(np.diff(az) >= -1e-9)  # 単調増加
    assert az[0] < 1.0
    assert az[-1] > 99.0
    # smoothstep なので中央付近の変化率が端より大きい
    mid_slope = az[105] - az[95]
    edge_slope = az[10] - az[0]
    assert mid_slope > edge_slope


def test_face_event_roundtrip():
    """よそ見 → 戻る、で face が 0 → target → 0 と推移する。"""
    away = MoveEvent("face_away", 10.0, 11.0, 30.0, 30.0, 1.0, 1.0, 0.0, 0.8)
    back = MoveEvent("face_back", 15.0, 17.0, 30.0, 30.0, 1.0, 1.0, 0.8, 0.0)
    traj = Trajectory(_plan_without_sway([away, back], init_az=30.0))

    az, _, face = traj.sample(np.array([5.0, 13.0, 30.0]))
    np.testing.assert_allclose(face, [0.0, 0.8, 0.0], atol=0.01)
    np.testing.assert_allclose(az, 30.0, atol=0.5)  # 位置は変わらない


def test_sway_is_bounded_and_deterministic():
    """揺らぎは指定振幅に収まり、同じシードで再現される。"""
    params = PlannerParams(seed=4, sway_az_deg=2.5, sway_dist_m=0.03)
    plan = Plan(duration=60.0, init_az=0.0, init_dist=1.0, params=params, events=[])

    t = np.linspace(0, 60, 1000)
    az1, dist1, _ = Trajectory(plan).sample(t)
    az2, dist2, _ = Trajectory(plan).sample(t)

    np.testing.assert_allclose(az1, az2)
    assert np.max(np.abs(az1)) <= 2.5 + 1e-6
    assert np.max(np.abs(dist1 - 1.0)) <= 0.03 + 1e-6
    # 完全静止ではない
    assert np.std(az1) > 0.1


def test_sample_clamps_out_of_range():
    """範囲外の時刻 (プレビューの端) でも端値でクランプされて落ちない。"""
    traj = Trajectory(_plan_without_sway([], duration=10.0, init_az=15.0))
    az, _, _ = traj.sample(np.array([-1.0, 100.0]))
    np.testing.assert_allclose(az, 15.0, atol=0.5)
