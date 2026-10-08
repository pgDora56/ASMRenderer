"""実 ASMR 音源 (バイノーラル録音) から話者の移動統計を推定する。

使い方:
    uv run python tools/analyze_reference.py <mp3 のあるフォルダ> <結果 JSON の出力先>

各フレーム (200ms 窓 / 100ms ホップ) で
  - ITD (GCC-PHAT) → 横方向の方位角推定 (前後は区別できない)
  - ILD, 全体レベル, 高域比率 (距離 / 向きの代理指標)
を取り、方位角の時系列から「移動」「滞留」を切り出して統計を出す。
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

SR = 16000
WIN = int(0.2 * SR)
HOP = int(0.1 * SR)
MAXLAG = int(0.0009 * SR)  # ±0.9ms
HEAD_ITD = 0.00066          # 横 90° での ITD の目安


def stream_frames(path: Path):
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf = np.zeros((0, 2), np.float32)
    chunk = HOP * 2 * 4 * 300  # 30s
    while True:
        raw = p.stdout.read(chunk)
        if not raw:
            break
        buf = np.concatenate([buf, np.frombuffer(raw, np.float32).reshape(-1, 2)])
        nf = (len(buf) - WIN) // HOP + 1
        if nf <= 0:
            continue
        idx = np.arange(nf)[:, None] * HOP + np.arange(WIN)[None, :]
        yield buf[idx]  # (nf, WIN, 2)
        buf = buf[nf * HOP:]


def analyze_file(path: Path) -> dict:
    w = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    lo = (freqs >= 100) & (freqs < 1000)
    hi = (freqs >= 2000) & (freqs < 6000)
    itd, ild, lvl, tilt = [], [], [], []
    for fr in stream_frames(path):
        L, R = fr[:, :, 0] * w, fr[:, :, 1] * w
        X, Y = np.fft.rfft(L, axis=1), np.fft.rfft(R, axis=1)
        G = X * np.conj(Y)
        G /= np.abs(G) + 1e-12
        cc = np.fft.irfft(G, n=WIN, axis=1)
        cc = np.concatenate([cc[:, -MAXLAG:], cc[:, : MAXLAG + 1]], axis=1)  # lag -MAXLAG..MAXLAG
        k = cc.argmax(axis=1)
        # 放物線補間でサブサンプル精度に
        km = np.clip(k - 1, 0, cc.shape[1] - 1)
        kp = np.clip(k + 1, 0, cc.shape[1] - 1)
        ym, y0, yp = cc[np.arange(len(k)), km], cc[np.arange(len(k)), k], cc[np.arange(len(k)), kp]
        denom = ym - 2 * y0 + yp
        frac = np.where(np.abs(denom) > 1e-9, 0.5 * (ym - yp) / np.where(denom == 0, 1, denom), 0.0)
        lag = (k - MAXLAG) + np.clip(frac, -1, 1)
        itd.append(lag / SR)
        rl, rr = np.sqrt((L**2).mean(1)) + 1e-9, np.sqrt((R**2).mean(1)) + 1e-9
        ild.append(20 * np.log10(rl / rr))
        lvl.append(20 * np.log10((rl + rr) / 2))
        P = np.abs(X) ** 2 + np.abs(Y) ** 2
        tilt.append(10 * np.log10(P[:, hi].sum(1) + 1e-12) - 10 * np.log10(P[:, lo].sum(1) + 1e-12))
    itd, ild, lvl, tilt = map(np.concatenate, (itd, ild, lvl, tilt))
    t = np.arange(len(itd)) * HOP / SR
    dur = t[-1]

    # 有音フレーム (上から方式: p95 - 25dB)
    thr = np.percentile(lvl, 95) - 25
    active = lvl > thr
    # 方位角: ITD ベース (正 = 左が先 → 左)。右を + にするため符号反転
    az_raw = -np.degrees(np.arcsin(np.clip(itd / HEAD_ITD, -1, 1)))
    az = az_raw.copy()
    az[~active] = np.nan
    # 無音区間は前の値で埋め、1.5 秒メディアンで平滑化
    idx = np.where(active, np.arange(len(az)), 0)
    np.maximum.accumulate(idx, out=idx)
    az_f = az[idx]
    az_f[np.isnan(az_f)] = 0.0
    az_s = ndimage.median_filter(az_f, size=15, mode="nearest")
    az_s = ndimage.uniform_filter1d(az_s, size=10, mode="nearest")

    # 移動検出: 1 秒あたりの変化量が閾値以上
    rate = np.abs(np.gradient(az_s, HOP / SR))
    moving = rate > 6.0
    moving = ndimage.binary_closing(moving, structure=np.ones(10))  # 1s 以内の途切れを結合
    lab, n = ndimage.label(moving)
    moves = []
    for i in range(1, n + 1):
        ii = np.where(lab == i)[0]
        s, e = ii[0], ii[-1]
        step = az_s[e] - az_s[s]
        if abs(step) < 12 or (e - s) < 3:
            continue
        seg_active = active[s:e + 1].mean()
        moves.append({
            "t0": round(float(t[s]), 1), "t1": round(float(t[e]), 1),
            "dur": round(float(t[e] - t[s]), 1), "step_deg": round(float(step), 1),
            "from": round(float(az_s[s]), 1), "to": round(float(az_s[e]), 1),
            "speech_ratio": round(float(seg_active), 2),
        })
    # 滞留区間ごとの統計
    bounds = [0.0] + [m["t1"] for m in moves] + [dur]
    starts = [0.0] + [m["t1"] for m in moves]
    ends = [m["t0"] for m in moves] + [dur]
    dwells = []
    for s0, e0 in zip(starts, ends):
        m = (t >= s0) & (t < e0) & active
        if m.sum() < 20:
            continue
        dwells.append({
            "t0": round(s0, 1), "t1": round(e0, 1), "len": round(e0 - s0, 1),
            "az_med": round(float(np.median(az_raw[m])), 1),
            "az_std": round(float(np.std(az_raw[m])), 1),
            "lvl_med": round(float(np.median(lvl[m])), 1),
            "tilt_med": round(float(np.median(tilt[m])), 1),
        })
    a = np.abs(az_raw[active])
    return {
        "file": path.name, "duration": round(dur, 1), "active_ratio": round(float(active.mean()), 3),
        "n_moves": len(moves), "moves_per_min": round(len(moves) / (dur / 60), 2),
        "az_occupancy": {"0-30": round(float((a < 30).mean()), 2), "30-60": round(float(((a >= 30) & (a < 60)).mean()), 2), "60+": round(float((a >= 60).mean()), 2)},
        "moves": moves, "dwells": dwells,
    }


def summarize(results: list[dict]) -> None:
    moves = [m for r in results for m in r["moves"]]
    dwells = [d for r in results for d in r["dwells"]]
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")
    def dist(name, v, fmt="{:.1f}"):
        print(f"  {name:28s} n={len(v):4d}  p10 {fmt.format(q(v,10))}  p50 {fmt.format(q(v,50))}  p90 {fmt.format(q(v,90))}  mean {fmt.format(np.mean(v) if len(v) else float('nan'))}")
    print("\n=== 全体サマリ ===")
    tot = sum(r["duration"] for r in results)
    print(f"  files {len(results)}  total {tot/3600:.1f}h  moves {len(moves)}  ({len(moves)/(tot/60):.2f}/min)")
    dist("move duration [s]", [m["dur"] for m in moves])
    dist("move |step| [deg]", [abs(m["step_deg"]) for m in moves])
    dist("move speech_ratio", [m["speech_ratio"] for m in moves], "{:.2f}")
    sr = np.array([m["speech_ratio"] for m in moves])
    if len(sr):
        print(f"  moves during speech (>0.7): {(sr>0.7).mean():.0%}   during pause (<0.3): {(sr<0.3).mean():.0%}   mixed: {((sr>=0.3)&(sr<=0.7)).mean():.0%}")
    dist("dwell length [s]", [d["len"] for d in dwells])
    dist("dwell |az_med| [deg]", [abs(d["az_med"]) for d in dwells])
    dist("dwell az_std (sway) [deg]", [d["az_std"] for d in dwells])
    if len(dwells) > 1:
        dl = [abs(b["lvl_med"] - a["lvl_med"]) for a, b in zip(dwells, dwells[1:]) if a["t1"] <= b["t0"]]
        dist("|Δlevel| between dwells [dB]", dl)
        dt = [abs(b["tilt_med"] - a["tilt_med"]) for a, b in zip(dwells, dwells[1:]) if a["t1"] <= b["t0"]]
        dist("|Δtilt| between dwells [dB]", dt)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in sorted(src.glob("*.mp3")):
        print(f"--- {f.name}", flush=True)
        r = analyze_file(f)
        results.append(r)
        (out_dir / (f.stem + ".json")).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  dur {r['duration']/60:.0f}min active {r['active_ratio']:.0%} moves {r['n_moves']} ({r['moves_per_min']}/min) occupancy {r['az_occupancy']}", flush=True)
    (out_dir / "_all.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    summarize(results)
