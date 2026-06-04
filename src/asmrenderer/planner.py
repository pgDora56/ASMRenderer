"""移動プランナー: 発話区間の解析結果から「話者がいつ・どこへ動くか」を決める。

設計方針:
- 移動は主に会話の切れ目 (ポーズ) の前後で起こす。喋り終わりにかけて動き始めたり、
  喋り出しと同時に移動が完了したりするよう、タイミングにジッタを入れる。
- たまに「喋りながらゆっくり移動」(walk_talk) を入れる。ASMR 感の核。
- 低頻度で「よそ見」(face_away): 向こうを向いて喋る → こもった声になる。
- 同じ位置にはある程度の時間 (dwell) 留まる。反復横跳びにならないように。
- 乱数はシード固定で再現可能。プランは JSON に保存でき、手で微調整して再レンダリングできる。

座標系:
- リスナーが原点、正面が方位角 0°。+90° が右、±180° が後ろ。
- プラン内の方位角は unwrap した連続値 (±180° を超えてよい)。補間が単純になる。
- face は 0 (こちら向き) 〜 1 (完全によそ見) の連続値。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .vad import SpeechAnalysis


@dataclass
class PlannerParams:
    """移動計画の生成パラメータ。時間は秒、角度は度。"""

    seed: int = 0

    # --- 位置移動 (ポーズトリガー) ---
    dwell_min_s: float = 25.0  # 同じ位置に留まる最短時間
    dwell_max_s: float = 110.0  # 同上の最長時間 (この範囲から一様サンプル)
    p_move_at_pause: float = 0.6  # dwell 経過後、ポーズで実際に移動する確率
    min_pause_s: float = 0.5  # 移動トリガーとみなす最小ポーズ長

    # --- 移動量 ---
    az_step_min_deg: float = 35.0
    az_step_max_deg: float = 150.0
    p_lean: float = 0.25  # 角度はほぼ変えず距離だけ変える「離れる/近づく」移動の割合
    walk_speed_deg_s: float = 60.0  # 移動アニメーションの角速度の目安

    # --- 喋りながら移動 ---
    walk_talk_rate_per_min: float = 0.25  # 期待回数 / 分 (≒4 分に 1 回)
    walk_talk_min_speech_s: float = 12.0  # この長さ以上の発話区間にだけ入れる
    walk_talk_dur_min_s: float = 5.0
    walk_talk_dur_max_s: float = 12.0

    # --- よそ見 ---
    face_away_rate_per_min: float = 0.12  # ≒8 分に 1 回。頻発すると落ち着きがなくなる
    face_away_hold_min_s: float = 2.0
    face_away_hold_max_s: float = 6.0

    # --- 揺らぎ (トラジェクトリ側で付与する微小な体の揺れ) ---
    sway_az_deg: float = 2.5
    sway_dist_m: float = 0.03


@dataclass
class MoveEvent:
    """1 回の状態遷移。t0〜t1 の間に from 値から to 値へ滑らかに補間される。"""

    kind: str  # "pause_jump" | "walk_talk" | "face_away" | "face_back"
    t0: float
    t1: float
    az0: float
    az1: float
    dist0: float
    dist1: float
    face0: float
    face1: float


@dataclass
class Plan:
    """移動計画の全体。JSON との相互変換が可能。"""

    duration: float
    init_az: float
    init_dist: float
    params: PlannerParams = field(default_factory=PlannerParams)
    events: list[MoveEvent] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "duration": self.duration,
                "init": {"az": round(self.init_az, 2), "dist": round(self.init_dist, 3)},
                "params": asdict(self.params),
                "events": [
                    {
                        "kind": e.kind,
                        "t0": round(e.t0, 3),
                        "t1": round(e.t1, 3),
                        "az0": round(e.az0, 2),
                        "az1": round(e.az1, 2),
                        "dist0": round(e.dist0, 3),
                        "dist1": round(e.dist1, 3),
                        "face0": round(e.face0, 3),
                        "face1": round(e.face1, 3),
                    }
                    for e in self.events
                ],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        obj = json.loads(text)
        params = PlannerParams(**obj.get("params", {}))
        events = [MoveEvent(**e) for e in obj["events"]]
        return cls(
            duration=obj["duration"],
            init_az=obj["init"]["az"],
            init_dist=obj["init"]["dist"],
            params=params,
            events=events,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        return cls.from_json(Path(path).read_text())


# --- 候補イベント (タイミングと移動量だけ決まっていて、位置はまだ未割り当て) ---
@dataclass
class _Candidate:
    kind: str
    t0: float
    t1: float
    az_step: float  # 角度変化の絶対量 (符号は割り当て時に決める)
    new_dist: float


def plan_movement(analysis: SpeechAnalysis, params: PlannerParams | None = None) -> Plan:
    """発話区間の解析結果から移動計画を生成する。"""
    params = params or PlannerParams()
    rng = np.random.default_rng(params.seed)
    duration = analysis.duration

    # 1. ポーズトリガーの位置移動候補を時系列順に生成 (dwell で間隔を制御)
    candidates = _gen_pause_candidates(analysis, params, rng)

    # 2. 喋りながら移動の候補を追加
    candidates += _gen_walk_talk_candidates(analysis, params, rng)

    # 3. 時系列順に整列し、重複・dwell 違反をグリーディに除去
    candidates.sort(key=lambda c: c.t0)
    accepted: list[_Candidate] = []
    for c in candidates:
        if accepted and c.t0 - accepted[-1].t1 < params.dwell_min_s:
            continue
        accepted.append(c)

    # 4. 初期位置を決め、候補に実際の方位角を順次割り当てる
    init_az = float(rng.uniform(-40, 40))
    init_dist = float(rng.uniform(0.6, 1.2))
    events = _assign_positions(accepted, init_az, init_dist, rng)

    # 5. よそ見イベントを、位置移動と重ならない発話中の時間帯に挿入
    events += _gen_face_events(analysis, events, params, rng, duration)
    events.sort(key=lambda e: e.t0)

    return Plan(
        duration=duration,
        init_az=init_az,
        init_dist=init_dist,
        params=params,
        events=events,
    )


def _gen_pause_candidates(
    analysis: SpeechAnalysis, params: PlannerParams, rng: np.random.Generator
) -> list[_Candidate]:
    """ポーズをトリガーにした位置移動候補を生成する。"""
    candidates = []
    last_move_end = 0.0
    dwell = rng.uniform(params.dwell_min_s, params.dwell_max_s)

    for ps, pe in analysis.pauses(min_len=params.min_pause_s):
        if ps - last_move_end < dwell:
            continue
        if rng.random() > params.p_move_at_pause:
            continue

        # 移動量を決める。「離れる/近づく」(lean) は角度小・距離変化メイン
        if rng.random() < params.p_lean:
            az_step = rng.uniform(5, 25)
        else:
            az_step = rng.uniform(params.az_step_min_deg, params.az_step_max_deg)
        new_dist = _sample_distance(rng)
        move_dur = float(np.clip(az_step / params.walk_speed_deg_s + 0.8, 1.2, 5.0))

        # タイミングのジッタ: ポーズ内 / 喋り終わりにかけて / 喋り出しにかけて
        r = rng.random()
        if r < 0.5:
            t0 = ps + rng.uniform(0.0, max(0.0, (pe - ps - move_dur) * 0.5))
        elif r < 0.75:
            t0 = ps - rng.uniform(0.3, 1.2)  # 語尾を言いながら動き出す
        else:
            t0 = pe - move_dur + rng.uniform(0.3, 1.5)  # 喋り出しの直後に移動が終わる
        t0 = max(0.0, t0)
        t1 = min(t0 + move_dur, analysis.duration)
        if t1 - t0 < 0.5:
            continue

        candidates.append(_Candidate("pause_jump", t0, t1, az_step, new_dist))
        last_move_end = t1
        dwell = rng.uniform(params.dwell_min_s, params.dwell_max_s)

    return candidates


def _gen_walk_talk_candidates(
    analysis: SpeechAnalysis, params: PlannerParams, rng: np.random.Generator
) -> list[_Candidate]:
    """喋りながらゆっくり移動する候補を、長い発話区間の中に生成する。"""
    candidates = []
    for ss, se in analysis.segments:
        seg_len = se - ss
        if seg_len < params.walk_talk_min_speech_s:
            continue
        # 区間長に比例した Poisson 回数だけ試みる
        n = rng.poisson(params.walk_talk_rate_per_min * seg_len / 60.0)
        for _ in range(n):
            dur = rng.uniform(params.walk_talk_dur_min_s, params.walk_talk_dur_max_s)
            if seg_len <= dur:
                continue
            t0 = ss + rng.uniform(0, seg_len - dur)
            # walk_talk はゆっくりなので角度は控えめ
            az_step = rng.uniform(30, 100)
            candidates.append(
                _Candidate("walk_talk", t0, t0 + dur, az_step, _sample_distance(rng))
            )
    return candidates


def _sample_distance(rng: np.random.Generator) -> float:
    """新しい距離 [m] をサンプルする。ASMR なので近距離を厚めに。"""
    r = rng.random()
    if r < 0.55:
        return float(rng.uniform(0.45, 0.9))  # 近い (ささやき距離)
    if r < 0.85:
        return float(rng.uniform(0.9, 1.6))  # ふつうの会話距離
    return float(rng.uniform(1.6, 2.4))  # 少し離れる


def _assign_positions(
    candidates: list[_Candidate],
    init_az: float,
    init_dist: float,
    rng: np.random.Generator,
) -> list[MoveEvent]:
    """候補に方位角の符号 (回る向き) を決めて、連続した位置遷移に変換する。"""
    events = []
    az, dist = init_az, init_dist
    for c in candidates:
        # 背後に長く留まらないよう、現在背後寄りなら正面方向へのバイアスをかける
        wrapped = ((az + 180) % 360) - 180
        if abs(wrapped) > 90 and rng.random() < 0.65:
            direction = -np.sign(wrapped) or 1.0
        else:
            direction = rng.choice([-1.0, 1.0])
        new_az = az + direction * c.az_step
        events.append(
            MoveEvent(
                kind=c.kind,
                t0=c.t0,
                t1=c.t1,
                az0=az,
                az1=float(new_az),
                dist0=dist,
                dist1=c.new_dist,
                face0=0.0,
                face1=0.0,
            )
        )
        az, dist = float(new_az), c.new_dist
    return events


def _gen_face_events(
    analysis: SpeechAnalysis,
    position_events: list[MoveEvent],
    params: PlannerParams,
    rng: np.random.Generator,
    duration: float,
) -> list[MoveEvent]:
    """よそ見 (face_away → face_back) のイベント対を生成する。

    位置移動イベントと重ならない発話中の時間帯にだけ入れる。
    """
    events: list[MoveEvent] = []
    pos_spans = [(e.t0 - 2.0, e.t1 + 2.0) for e in position_events]  # 前後 2 秒のマージン
    last_face_end = -1e9

    for ss, se in analysis.segments:
        seg_len = se - ss
        if seg_len < 6.0:
            continue
        n = rng.poisson(params.face_away_rate_per_min * seg_len / 60.0)
        for _ in range(n):
            turn_in = rng.uniform(0.8, 1.5)  # 向こうを向く所要時間
            hold = rng.uniform(params.face_away_hold_min_s, params.face_away_hold_max_s)
            turn_back = rng.uniform(1.0, 2.0)
            total = turn_in + hold + turn_back
            if seg_len <= total + 1.0:
                continue
            t0 = ss + rng.uniform(0.5, seg_len - total - 0.5)
            t_end = t0 + total
            # 位置移動・既存のよそ見との重複を避ける
            if any(t0 < pe and t_end > p0 for p0, pe in pos_spans):
                continue
            if t0 - last_face_end < 30.0:  # よそ見同士は最低 30 秒空ける
                continue

            # この時刻の位置を求める (位置イベントは face を変えないので az/dist は直前の値)
            az, dist = _state_at(position_events, t0)
            face_target = float(rng.uniform(0.5, 1.0))
            events.append(
                MoveEvent("face_away", t0, t0 + turn_in, az, az, dist, dist, 0.0, face_target)
            )
            events.append(
                MoveEvent(
                    "face_back",
                    t0 + turn_in + hold,
                    min(t_end, duration),
                    az,
                    az,
                    dist,
                    dist,
                    face_target,
                    0.0,
                )
            )
            last_face_end = t_end
    return events


def _state_at(position_events: list[MoveEvent], t: float) -> tuple[float, float]:
    """時刻 t における (az, dist) を位置イベント列から求める。"""
    az, dist = None, None
    for e in position_events:
        if e.t1 <= t:
            az, dist = e.az1, e.dist1
        elif e.t0 <= t < e.t1:
            # 遷移中 (通常は重複回避で起こらないが念のため線形近似)
            u = (t - e.t0) / (e.t1 - e.t0)
            az, dist = e.az0 + (e.az1 - e.az0) * u, e.dist0 + (e.dist1 - e.dist0) * u
        else:
            break
    if az is None:
        # 最初の位置イベントより前 → イベント列の先頭の from 値、なければ呼び出し側の初期値
        if position_events:
            return position_events[0].az0, position_events[0].dist0
        return 0.0, 1.0
    return az, dist
