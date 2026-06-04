"""トラジェクトリ: 移動計画 (Plan) を連続的な位置カーブに変換する。

メモリ節約のため、カーブは制御レート (デフォルト 100Hz) で保持し、
レンダラーが必要な区間だけサンプルレートに線形補間して使う。
2 時間の音源でも制御点は 72 万点程度で済む。
"""

from __future__ import annotations

import numpy as np

from .planner import Plan


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """滑らかな加減速カーブ (3u^2 - 2u^3)。歩き出し・止まり際が自然になる。"""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Trajectory:
    """時刻 t → (方位角 deg, 距離 m, よそ見度 0-1) を返す連続カーブ。"""

    def __init__(self, plan: Plan, control_rate: float = 100.0):
        self.plan = plan
        self.control_rate = control_rate

        n = int(plan.duration * control_rate) + 2
        self.t = np.arange(n) / control_rate
        # まず初期状態で埋め、イベントを時系列順に上書きしていく
        self.az = np.full(n, plan.init_az, dtype=np.float64)
        self.dist = np.full(n, plan.init_dist, dtype=np.float64)
        self.face = np.zeros(n, dtype=np.float64)

        for e in sorted(plan.events, key=lambda e: e.t0):
            i0 = min(n - 1, max(0, int(e.t0 * control_rate)))
            i1 = min(n - 1, max(i0 + 1, int(e.t1 * control_rate)))
            u = _smoothstep((self.t[i0:i1] - e.t0) / max(e.t1 - e.t0, 1e-6))
            # 遷移区間: smoothstep 補間。以降はイベント終了値でホールド
            self.az[i0:i1] = e.az0 + (e.az1 - e.az0) * u
            self.az[i1:] = e.az1
            self.dist[i0:i1] = e.dist0 + (e.dist1 - e.dist0) * u
            self.dist[i1:] = e.dist1
            self.face[i0:i1] = e.face0 + (e.face1 - e.face0) * u
            self.face[i1:] = e.face1

        self._add_sway()

    def _add_sway(self) -> None:
        """微小な体の揺れを加える。完全静止だと不自然なので、ゆっくりした正弦波を重ねる。"""
        p = self.plan.params
        if p.sway_az_deg <= 0:
            return
        # シードから決定論的に位相・周期を決める (プラン本体の乱数列とは独立)
        rng = np.random.default_rng(p.seed + 1)
        p1, p2 = rng.uniform(9, 15), rng.uniform(20, 35)
        ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
        self.az += p.sway_az_deg * (
            0.6 * np.sin(2 * np.pi * self.t / p1 + ph1)
            + 0.4 * np.sin(2 * np.pi * self.t / p2 + ph2)
        )
        p3 = rng.uniform(11, 19)
        self.dist += p.sway_dist_m * np.sin(2 * np.pi * self.t / p3 + rng.uniform(0, 2 * np.pi))
        np.clip(self.dist, 0.3, None, out=self.dist)

    def sample(self, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """任意の時刻列に対して (az, dist, face) を線形補間で返す。範囲外は端値でクランプ。"""
        az = np.interp(t_query, self.t, self.az)
        dist = np.interp(t_query, self.t, self.dist)
        face = np.interp(t_query, self.t, self.face)
        return az, dist, face
