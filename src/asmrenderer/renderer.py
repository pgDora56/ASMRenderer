"""バイノーラルレンダラー: トラジェクトリに沿って音源を空間化する。

HRTF は使わず、パラメトリックな手がかりの組み合わせで定位させる (v0.1):

- ITD: 両耳それぞれへの伝搬遅延を幾何学的に計算し、サンプル単位の
  可変フラクショナル遅延として適用。移動時はドップラー風の自然な揺らぎも出る。
- ILD + 距離減衰: 両耳それぞれへの距離に応じた 1/r ゲイン。
- 頭部シャドウ: 音源と反対側の耳の高域を減衰 (STFT ベースの時変フィルタ)。
- 背後キュー: 背後にいるとき高域をわずかにディップさせる (簡易的な前後の手がかり)。
- よそ見: 声の指向性を模して、高域カット + レベル低下。残響は減らさないので
  「向こうを向いて喋っている」感じになる。
- 距離感: 合成 IR による軽いルームリバーブ。遠い / よそ見ほど wet を増やす。

処理はチャンク (フレームのバッチ) 単位で行い、2 時間級の音源でも
メモリに乗る範囲でベクトル演算する。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from scipy import signal

SPEED_OF_SOUND = 343.0  # [m/s]


@dataclass
class RendererParams:
    head_radius_m: float = 0.0875  # 平均的な頭部半径
    ref_dist_m: float = 1.0  # このゲイン基準距離で 1 倍
    near_clip_m: float = 0.25  # これ以上近づいてもゲインを増やさない (爆音防止)

    # 広帯域 ILD: 真横 (|sin az| = 1) で近い耳と遠い耳の差が lateral_ild_db になる。
    # 耳間距離だけの 1/r 差 (~2dB) では実録音 (中央値 6dB、真横近距離で 13dB 超) に
    # 遠く及ばず移動が小さく聞こえるため、明示的に付与する。近距離ほど増える。
    lateral_ild_db: float = 10.0
    ild_near_boost: float = 0.5  # dist <= ild_near_dist で ILD が (1 + boost) 倍
    ild_near_dist: float = 0.4
    ild_far_dist: float = 1.0  # これ以遠では boost なし

    # 頭部シャドウ: 反対側の耳の高域減衰の深さ (0-1) と帯域
    shadow_depth: float = 0.55
    shadow_f0: float = 700.0
    shadow_f1: float = 4500.0
    shadow_broadband_db: float = -2.5  # 高域以外もわずかに下がる

    # 背後キュー: 背後にいるときの高域ディップ
    behind_dip: float = 0.35
    behind_f0: float = 3000.0
    behind_f1: float = 8000.0

    # よそ見: 高域カットの深さとレベル低下
    face_hf_cut: float = 0.85
    face_f0: float = 1000.0
    face_f1: float = 5000.0
    face_level_db: float = -5.0

    # リバーブ
    reverb_t60_s: float = 0.35
    reverb_predelay_s: float = 0.012
    reverb_lp_hz: float = 4000.0
    wet_base: float = 0.05  # 常時の wet 量
    wet_dist: float = 0.10  # 距離による wet 増加 (dist=2.5m で最大)
    wet_face: float = 0.18  # よそ見による wet 増加

    # STFT 設定 (hop = frame // 2 固定。Hann 窓の OLA がちょうど 1 になる)
    frame: int = 1024
    batch_frames: int = 4096  # 1 バッチで処理するフレーム数 (~44 秒 @48kHz)


def _freq_ramp(freqs: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """f0 以下で 0、f1 以上で 1 になる滑らかな周波数重み。"""
    u = np.clip((freqs - f0) / (f1 - f0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _make_reverb_ir(sr: int, params: RendererParams) -> tuple[np.ndarray, np.ndarray]:
    """合成ルーム IR (L/R 非相関) を作る。指数減衰ノイズ + ローパス。"""
    rng = np.random.default_rng(20260605)  # IR は常に同一 (出力の再現性のため固定シード)
    n = int(params.reverb_t60_s * sr)
    t = np.arange(n) / sr
    # T60 で -60dB になる指数減衰
    env = np.exp(-6.91 * t / params.reverb_t60_s)
    sos = signal.butter(2, params.reverb_lp_hz, fs=sr, output="sos")
    irs = []
    for _ in range(2):
        ir = signal.sosfilt(sos, rng.standard_normal(n) * env)
        ir /= np.sqrt(np.sum(ir**2))  # エネルギー正規化: wet ゲインがそのまま比率になる
        pre = np.zeros(int(params.reverb_predelay_s * sr))
        irs.append(np.concatenate([pre, ir]).astype(np.float32))
    return irs[0], irs[1]


def render(
    audio: np.ndarray,
    sr: int,
    trajectory,
    params: RendererParams | None = None,
    t_offset: float = 0.0,
    progress: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """モノラル音声をトラジェクトリに沿ってバイノーラル化する。

    Args:
        audio: モノラル float32 (n,)
        t_offset: audio の先頭が対応するトラジェクトリ上の時刻。プレビュー用の
            部分レンダリングで使う。
        normalize: True ならピークを -1dBFS に正規化する。後段で BGM と
            ミックスする場合は False にして、ミックス後に正規化する。
    Returns:
        ステレオ float32 (n, 2)。
    """
    params = params or RendererParams()
    frame, hop = params.frame, params.frame // 2
    n = len(audio)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    n_frames = max(1, (n - 1) // hop + 1)
    out = np.zeros((n_frames * hop + frame, 2), dtype=np.float64)

    window = signal.get_window("hann", frame, fftbins=True)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    # 周波数重みは全フレーム共通なので事前計算
    w_shadow = _freq_ramp(freqs, params.shadow_f0, params.shadow_f1)
    w_behind = _freq_ramp(freqs, params.behind_f0, params.behind_f1)
    w_face = _freq_ramp(freqs, params.face_f0, params.face_f1)

    a = params.head_radius_m
    margin = int(0.05 * sr)  # 最大伝搬遅延 (~3m で 9ms) より十分大きい読み出しマージン

    n_batches = (n_frames + params.batch_frames - 1) // params.batch_frames
    for b in range(n_batches):
        k0 = b * params.batch_frames
        k1 = min(k0 + params.batch_frames, n_frames)
        rs = k0 * hop  # このバッチが必要とするサンプル領域 [rs, re)
        re = (k1 - 1) * hop + frame

        # --- 領域内の per-sample 位置・幾何を計算 ---
        idx = np.arange(rs, re, dtype=np.float64)
        t = idx / sr + t_offset
        az_deg, dist, face = trajectory.sample(t)
        theta = np.deg2rad(az_deg)
        # リスナー原点・正面 +y。音源位置と各耳 (x = ∓a) への距離
        sx, sy = dist * np.sin(theta), dist * np.cos(theta)
        r_l = np.hypot(sx + a, sy)
        r_r = np.hypot(sx - a, sy)

        # --- per-sample の遅延 + ゲイン (ITD / ILD / 距離減衰 / ドップラー) ---
        src0 = max(0, rs - margin)
        src_idx = np.arange(src0, min(n, re), dtype=np.float64)
        src = audio[int(src0) : min(n, re)]
        # 広帯域 ILD (per-sample なので移動中も滑らか)。近距離ほど大きい
        near = np.clip(
            (params.ild_far_dist - dist) / (params.ild_far_dist - params.ild_near_dist), 0.0, 1.0
        )
        ild_db = params.lateral_ild_db * (1.0 + params.ild_near_boost * near)
        sin_t = np.sin(theta)
        ears = []
        for r_ear, s_near in ((r_l, -sin_t), (r_r, sin_t)):
            delay = r_ear / SPEED_OF_SOUND * sr  # [サンプル]
            gain = params.ref_dist_m / np.maximum(r_ear, params.near_clip_m)
            # 音源側の耳は +ild/2、反対側は -ild/2 (s_near は音源側で正)
            gain = gain * 10 ** (ild_db * s_near / 40.0)
            # 入力を「遅延分だけ過去」の位置から線形補間で読む = 可変フラクショナル遅延
            ear = np.interp(idx - delay, src_idx, src, left=0.0, right=0.0) * gain
            ears.append(ear)

        # --- フレームごとのスペクトル整形 (頭部シャドウ / 背後 / よそ見) ---
        n_bf = k1 - k0
        centers = (np.arange(n_bf) * hop + frame // 2).clip(0, re - rs - 1)
        az_c = theta[centers]
        face_c = face[centers]
        sin_c, cos_c = np.sin(az_c), np.cos(az_c)
        behind = np.maximum(0.0, -cos_c)  # 背後度 0-1
        # よそ見の共通フィルタ: 高域カット + レベル低下
        g_face = (1.0 - params.face_hf_cut * face_c[:, None] * w_face) * 10 ** (
            params.face_level_db * face_c[:, None] / 20
        )
        g_behind = 1.0 - params.behind_dip * behind[:, None] * w_behind

        for ch, ear in enumerate(ears):
            # 音源と反対側の耳ほどシャドウが深い (左耳 ch=0 は音源が右 sin>0 のとき)
            s_ear = np.maximum(0.0, sin_c if ch == 0 else -sin_c)
            g_shadow = (1.0 - params.shadow_depth * s_ear[:, None] * w_shadow) * 10 ** (
                params.shadow_broadband_db * s_ear[:, None] / 20
            )
            gains = g_shadow * g_behind * g_face  # (n_bf, bins)

            # フレーム化 → FFT → ゲイン適用 → 逆 FFT
            pad = np.zeros(max(0, (n_bf - 1) * hop + frame - len(ear)))
            frames = np.lib.stride_tricks.sliding_window_view(
                np.concatenate([ear, pad]), frame
            )[::hop][:n_bf]
            spec = np.fft.rfft(frames * window, axis=1)
            shaped = np.fft.irfft(spec * gains, n=frame, axis=1)

            # OLA: hop = frame/2 なので偶数 / 奇数フレームはそれぞれ隙間なく並ぶ
            for parity in (0, 1):
                local = np.arange(n_bf)[(np.arange(k0, k1) % 2) == parity]
                if len(local) == 0:
                    continue
                start = (k0 + local[0]) * hop
                block = shaped[local].ravel()
                out[start : start + len(block), ch] += block

        if progress:
            print(f"\r  render: {100 * (b + 1) // n_batches}%", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)

    out = out[:n]

    # --- リバーブ: dry 全体を 1 回畳み込み、per-sample の wet ゲインで混ぜる ---
    ir_l, ir_r = _make_reverb_ir(sr, params)
    t_all = np.arange(n, dtype=np.float64) / sr + t_offset
    _, dist_all, face_all = trajectory.sample(t_all)
    wet_gain = (
        params.wet_base
        + params.wet_dist * np.clip((dist_all - 0.5) / 2.0, 0.0, 1.0)
        + params.wet_face * face_all
    )
    for ch, ir in enumerate((ir_l, ir_r)):
        wet = signal.oaconvolve(audio, ir)[:n]
        out[:, ch] += wet * wet_gain

    # --- ピーク正規化 (-1 dBFS) ---
    if normalize:
        peak = np.max(np.abs(out))
        if peak > 0:
            out *= 10 ** (-1.0 / 20) / peak
    return out.astype(np.float32)
