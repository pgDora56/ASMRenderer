"""CLI エントリポイント。

サブコマンド:
- run:      解析 → プラン生成 → レンダリングを一気に実行 (通常はこれだけでよい)
- analyze:  発話区間検出だけ実行して JSON に保存
- plan:     解析結果 JSON から移動計画 JSON を生成
- render:   音声 + 計画 JSON からレンダリング (プランを手で微調整した後の再実行用)
- separate: 音源分離 (声 / BGM) だけ実行してキャッシュを作る

プランは JSON の中間成果物として残すので、シードを変えて引き直したり、
特定の移動だけ手で書き換えて render し直すワークフローを想定している。

BGM 入りの配信音源は `--separate` を付けると、声だけを移動させて
BGM は音量を下げて静止ミックスする (要 `.[separate]` extra)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .audio_io import load_audio, load_audio_mono, save_audio
from .planner import Plan, PlannerParams, plan_movement
from .renderer import RendererParams, render
from .separation import separate_vocals
from .trajectory import Trajectory
from .vad import SpeechAnalysis, detect_speech


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="asmrenderer",
        description="音声を「話者が周囲を移動する」ASMR 風バイノーラル音源に変換する",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="解析からレンダリングまで一気に実行")
    p_run.add_argument("input", help="入力音声ファイル (wav/flac/mp3 など)")
    p_run.add_argument("-o", "--output", help="出力ファイル (デフォルト: <入力名>_asmr.wav)")
    p_run.add_argument("--seed", type=int, default=0, help="移動計画の乱数シード")
    p_run.add_argument("--vad", choices=["energy", "silero"], default="energy")
    p_run.add_argument("--plan-out", help="生成した移動計画 JSON の保存先 (デフォルト: <出力名>.plan.json)")
    _add_separate_args(p_run)
    _add_preview_args(p_run)

    p_an = sub.add_parser("analyze", help="発話区間検出のみ実行")
    p_an.add_argument("input")
    p_an.add_argument("-o", "--output", required=True, help="解析結果 JSON の保存先")
    p_an.add_argument("--vad", choices=["energy", "silero"], default="energy")
    _add_separate_args(p_an)

    p_pl = sub.add_parser("plan", help="解析結果 JSON から移動計画を生成")
    p_pl.add_argument("segments", help="analyze の出力 JSON")
    p_pl.add_argument("-o", "--output", required=True, help="移動計画 JSON の保存先")
    p_pl.add_argument("--seed", type=int, default=0)

    p_re = sub.add_parser("render", help="音声と移動計画 JSON からレンダリング")
    p_re.add_argument("input")
    p_re.add_argument("plan", help="移動計画 JSON")
    p_re.add_argument("-o", "--output", required=True)
    _add_separate_args(p_re)
    _add_preview_args(p_re)

    p_sep = sub.add_parser("separate", help="音源分離 (声/BGM) のみ実行")
    p_sep.add_argument("input")
    p_sep.add_argument("--force-separate", action="store_true", help="キャッシュを無視して分離し直す")

    args = parser.parse_args(argv)
    return {
        "run": _cmd_run,
        "analyze": _cmd_analyze,
        "plan": _cmd_plan,
        "render": _cmd_render,
        "separate": _cmd_separate,
    }[args.command](args)


def _add_preview_args(parser: argparse.ArgumentParser) -> None:
    """長尺音源の一部だけ試しに書き出すためのオプション。"""
    parser.add_argument("--start", type=float, default=0.0, help="レンダリング開始位置 [秒]")
    parser.add_argument("--duration", type=float, help="レンダリングする長さ [秒] (省略時は最後まで)")


def _add_separate_args(parser: argparse.ArgumentParser) -> None:
    """音源分離まわりのオプション。"""
    parser.add_argument(
        "--separate",
        action="store_true",
        help="声と BGM を分離し、声だけを移動させる (BGM は静止ミックス。要 .[separate])",
    )
    parser.add_argument(
        "--bgm-gain-db",
        type=float,
        default=-6.0,
        help="静止ミックスする BGM のゲイン [dB] (デフォルト: -6)",
    )
    parser.add_argument("--force-separate", action="store_true", help="分離キャッシュを無視してやり直す")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_sources(args: argparse.Namespace) -> tuple[np.ndarray, int, np.ndarray | None]:
    """入力を読み込む。--separate 時は (声モノラル, sr, BGM ステレオ) を返す。

    分離結果はキャッシュされるので、2 回目以降は分離処理が走らない。
    """
    if getattr(args, "separate", False):
        vocals_path, bgm_path = separate_vocals(args.input, force=args.force_separate)
        _log(f"分離済み stems を使用: {vocals_path.parent}")
        vocals, sr = load_audio_mono(vocals_path)
        bgm, bgm_sr = load_audio(bgm_path)
        if bgm_sr != sr:
            raise SystemExit(f"stems のサンプルレートが不一致です: {sr} != {bgm_sr}")
        return vocals, sr, bgm
    audio, sr = load_audio_mono(args.input)
    return audio, sr, None


def _cmd_run(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_asmr.wav")
    plan_path = Path(args.plan_out) if args.plan_out else out_path.with_suffix(".plan.json")

    _log(f"読み込み: {in_path}")
    audio, sr, bgm = _load_sources(args)
    _log(f"  {len(audio) / sr / 60:.1f} 分, {sr} Hz")

    _log(f"発話区間検出 ({args.vad}) ...")
    analysis = detect_speech(audio, sr, method=args.vad)
    _log(f"  発話区間 {len(analysis.segments)} 個, "
         f"移動候補ポーズ {len(analysis.pauses(min_len=0.5))} 個")
    if not analysis.segments:
        _log("  警告: 発話が検出できませんでした。移動のない静的な音源になります。")

    plan = plan_movement(analysis, PlannerParams(seed=args.seed))
    plan.save(plan_path)
    _summarize_plan(plan)
    _log(f"  プラン保存: {plan_path} (編集して `render` で再実行できます)")

    _render_and_save(audio, sr, plan, out_path, args.start, args.duration, bgm, args.bgm_gain_db)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    audio, sr, _ = _load_sources(args)
    analysis = detect_speech(audio, sr, method=args.vad)
    analysis.save(args.output)
    _log(f"発話区間 {len(analysis.segments)} 個 → {args.output}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    analysis = SpeechAnalysis.load(args.segments)
    plan = plan_movement(analysis, PlannerParams(seed=args.seed))
    plan.save(args.output)
    _summarize_plan(plan)
    _log(f"プラン保存: {args.output}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    audio, sr, bgm = _load_sources(args)
    plan = Plan.load(args.plan)
    _render_and_save(
        audio, sr, plan, Path(args.output), args.start, args.duration, bgm, args.bgm_gain_db
    )
    return 0


def _cmd_separate(args: argparse.Namespace) -> int:
    vocals_path, bgm_path = separate_vocals(args.input, force=args.force_separate)
    _log(f"声:  {vocals_path}")
    _log(f"BGM: {bgm_path}")
    return 0


def _summarize_plan(plan: Plan) -> None:
    kinds: dict[str, int] = {}
    for e in plan.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    _log("移動計画: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))


def _render_and_save(
    audio: np.ndarray,
    sr: int,
    plan: Plan,
    out_path: Path,
    start: float,
    duration: float | None,
    bgm: np.ndarray | None = None,
    bgm_gain_db: float = -6.0,
) -> None:
    # プレビュー用の部分レンダリング: 音声を切り出し、トラジェクトリには絶対時刻を渡す
    i0 = int(start * sr)
    i1 = int((start + duration) * sr) if duration else len(audio)
    clip = audio[i0:i1]
    if len(clip) == 0:
        raise SystemExit("指定範囲に音声がありません")

    traj = Trajectory(plan)
    _log(f"レンダリング: {len(clip) / sr / 60:.1f} 分 ...")
    # BGM をミックスする場合は、声の空間化結果と BGM の相対バランスを保つため
    # 正規化をミックス後に回す
    out = render(clip, sr, traj, RendererParams(), t_offset=start, progress=True,
                 normalize=bgm is None)

    if bgm is not None:
        out = _mix_bgm(out, bgm[i0:i1], bgm_gain_db)

    save_audio(out_path, out, sr)
    _log(f"出力: {out_path}")


def _mix_bgm(spatial: np.ndarray, bgm: np.ndarray, gain_db: float) -> np.ndarray:
    """空間化した声に BGM を静止ステレオでミックスし、ピークを -1dBFS に正規化する。"""
    n = len(spatial)
    mixed = spatial.astype(np.float64)
    m = min(n, len(bgm))
    # BGM がモノラルならステレオに複製
    bgm2 = bgm if bgm.ndim == 2 else np.repeat(bgm[:, None], 2, axis=1)
    mixed[:m] += bgm2[:m, :2] * 10 ** (gain_db / 20)

    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed *= 10 ** (-1.0 / 20) / peak
    return mixed.astype(np.float32)


if __name__ == "__main__":
    sys.exit(main())
