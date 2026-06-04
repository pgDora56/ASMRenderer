"""移動プランナーのテスト。"""

from asmrenderer.planner import Plan, PlannerParams, plan_movement


def test_deterministic(long_analysis):
    """同じシードなら同じプランが出る。"""
    p1 = plan_movement(long_analysis, PlannerParams(seed=7))
    p2 = plan_movement(long_analysis, PlannerParams(seed=7))
    assert p1.to_json() == p2.to_json()


def test_different_seeds_differ(long_analysis):
    p1 = plan_movement(long_analysis, PlannerParams(seed=1))
    p2 = plan_movement(long_analysis, PlannerParams(seed=2))
    assert p1.to_json() != p2.to_json()


def test_position_moves_exist_and_spaced(long_analysis):
    """10 分の音源で位置移動が複数回発生し、dwell 制約を守っている。"""
    params = PlannerParams(seed=0)
    plan = plan_movement(long_analysis, params)

    pos = [e for e in plan.events if e.kind in ("pause_jump", "walk_talk")]
    # 10 分なら数回〜十数回は動くはず (dwell 25-110s なので上限 ~24 回)
    assert 3 <= len(pos) <= 24

    # 位置移動同士は dwell_min 以上空いている
    for prev, cur in zip(pos, pos[1:]):
        assert cur.t0 - prev.t1 >= params.dwell_min_s - 1e-6


def test_position_continuity(long_analysis):
    """位置移動イベントは前のイベントの終了位置から始まる (テレポートしない)。"""
    plan = plan_movement(long_analysis, PlannerParams(seed=3))
    pos = [e for e in plan.events if e.kind in ("pause_jump", "walk_talk")]

    az, dist = plan.init_az, plan.init_dist
    for e in pos:
        assert e.az0 == az
        assert e.dist0 == dist
        az, dist = e.az1, e.dist1


def test_events_within_duration(long_analysis):
    plan = plan_movement(long_analysis, PlannerParams(seed=5))
    for e in plan.events:
        assert 0.0 <= e.t0 < e.t1 <= long_analysis.duration + 1e-6


def test_face_events_paired(long_analysis):
    """よそ見は必ず away → back の対になっていて、face が 0 に戻る。"""
    # よそ見の頻度を上げて確実に発生させる
    params = PlannerParams(seed=11, face_away_rate_per_min=1.0)
    plan = plan_movement(long_analysis, params)

    aways = [e for e in plan.events if e.kind == "face_away"]
    backs = [e for e in plan.events if e.kind == "face_back"]
    assert len(aways) == len(backs)
    assert len(aways) >= 1
    for away, back in zip(aways, backs):
        assert away.face1 > 0.4
        assert back.face0 == away.face1
        assert back.face1 == 0.0
        assert away.t1 <= back.t0  # away の後に back


def test_face_events_not_overlapping_moves(long_analysis):
    """よそ見が位置移動と重ならない。"""
    params = PlannerParams(seed=13, face_away_rate_per_min=1.0)
    plan = plan_movement(long_analysis, params)
    pos = [e for e in plan.events if e.kind in ("pause_jump", "walk_talk")]
    faces = [e for e in plan.events if e.kind.startswith("face_")]
    for f in faces:
        for p in pos:
            assert f.t1 <= p.t0 or f.t0 >= p.t1


def test_no_moves_without_speech():
    """発話ゼロならイベントも出ない (落ちずに静的プランになる)。"""
    from asmrenderer.vad import SpeechAnalysis

    plan = plan_movement(SpeechAnalysis(duration=60.0, segments=[]), PlannerParams(seed=0))
    assert plan.events == []


def test_json_roundtrip(long_analysis):
    plan = plan_movement(long_analysis, PlannerParams(seed=9))
    restored = Plan.from_json(plan.to_json())
    assert restored.duration == plan.duration
    assert len(restored.events) == len(plan.events)
    assert restored.to_json() == plan.to_json()
