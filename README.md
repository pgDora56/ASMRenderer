# ASMRenderer

雑談配信などの長尺音声 (1〜2 時間想定) を、**話者が自分の周りを移動しながら喋っているような ASMR 風バイノーラル音源**に変換する CLI ツール。

ヘッドホン / イヤホンでの聴取を前提とする。

## コンセプト

- 移動は主に**会話の切れ目の前後**で起こる (語尾を言いながら動き出す、喋り出しと同時に移動が終わる、など)
- たまに**喋りながらゆっくり移動**する (ASMR 感の核)
- 低頻度で**よそ見**をする (向こうの物を取りながら喋る → 声がこもる)
- 同じ位置にしばらく留まり、反復横跳びにはならない
- 距離も変わる (ささやき距離 〜 少し離れた位置)

## セットアップ

```sh
uv sync
```

## 使い方

```sh
# 基本: これだけで解析 → 移動計画 → レンダリングまで実行
uv run asmrenderer run input.wav -o output.wav

# シードを変えると別の移動パターンになる
uv run asmrenderer run input.wav -o output.wav --seed 7

# 長尺音源はまず一部だけプレビューして雰囲気を確認するとよい
uv run asmrenderer run input.wav -o preview.wav --start 600 --duration 60

# BGM 入りの配信音源は --separate を推奨: 声だけが移動し、BGM は静止ミックスされる
uv pip install -e '.[separate]'
uv run asmrenderer run input.wav -o output.wav --separate

# BGM のミックス量を変える (デフォルト -6dB)
uv run asmrenderer run input.wav -o output.wav --separate --bgm-gain-db -10

# 分離だけ先に済ませておくこともできる (結果は <入力名>_stems/ にキャッシュされる)
uv run asmrenderer separate input.wav

# 分離を使わない場合、BGM 入り音源の VAD には silero を推奨 (要 extra)
uv pip install -e '.[silero]'
uv run asmrenderer run input.wav -o output.wav --vad silero
```

### YouTube の URL から直接変換

yt-dlp と ffmpeg が PATH にあれば、URL を渡すだけで取得 → 変換まで一気に実行できる:

```sh
# 取得した音声と出力は ~/Music/ASMRenderer/ に置かれる (--work-dir で変更可)
uv run asmrenderer yt "https://www.youtube.com/watch?v=XXXX" --separate

# 長尺配信は wav にすると 1GB 超になり全編をメモリに読むので、まず範囲を切って試す
uv run asmrenderer yt "https://www.youtube.com/watch?v=XXXX" --start 600 --duration 600 --separate --seed 7
```

ダウンロードと wav 変換はキャッシュされるので、同じ URL・同じ範囲ならシードを変えて何度でもすぐ引き直せる。
`--start` / `--duration` は wav の切り出しに使われ、出力は切り出した範囲の全編になる
(`run` の部分レンダリングとは違い、プランも切り出し範囲に対して作られる)。

### 移動計画の手動調整

`run` は移動計画を `<出力名>.plan.json` に保存する。気に入らない移動だけ JSON を直接編集して、再レンダリングできる:

```sh
uv run asmrenderer render input.wav output.plan.json -o output_v2.wav
```

ステップごとに分けて実行することもできる:

```sh
uv run asmrenderer analyze input.wav -o segments.json
uv run asmrenderer plan segments.json -o plan.json --seed 3
uv run asmrenderer render input.wav plan.json -o output.wav
```

## アーキテクチャ

```
音声 ─→ [separation] ─→ 声  ─→ [vad] ─→ [planner] 移動計画 (JSON) ─→ [renderer] バイノーラル化 ─┐
       (--separate 時)  └─→ BGM ────────────────────────────────→ 静止ステレオでミックス ──┴─→ ステレオ出力
```

| モジュール | 役割 |
|---|---|
| `separation.py` | 声 / BGM の音源分離 (Demucs)。結果は `<入力名>_stems/` にキャッシュ |
| `vad.py` | 発話区間 / ポーズの検出。energy (依存なし) と silero (BGM に強い) |
| `planner.py` | 「いつ・どこへ動くか」を確率的に決める。シード固定で再現可能 |
| `trajectory.py` | 計画を連続的な位置カーブに変換 (smoothstep 補間 + 微小な体の揺れ) |
| `renderer.py` | ITD / ILD / 距離減衰 / 頭部シャドウ / よそ見フィルタ / ルームリバーブ |

### Windows 11 での利用

本体はピュア Python なのでそのまま動く。音源分離を速くしたい場合:

1. NVIDIA GPU があるなら、demucs より先に CUDA 版 PyTorch を入れる
   ```sh
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -e '.[separate]'
   ```
2. GPU なし (CPU) でも動くが、2 時間音源の分離には相応の時間がかかる。
   分離結果はキャッシュされるので、一度 `separate` を流しておけば
   以降のシード変え・プラン調整は分離なしで高速に回せる

注意: Demucs の出力は 44.1kHz 固定 (htdemucs モデル準拠)。入力が 48kHz でも
`--separate` 時の出力は 44.1kHz になる。

### 座標系

- リスナーが原点、正面が方位角 0°。+90° が右、±180° が後ろ
- プラン内の方位角は unwrap した連続値 (±180° を超えてよい)
- `face`: 0 = こちらを向いている 〜 1 = 完全によそ見

### レンダリングの空間キュー (v0.1: パラメトリック方式)

- **ITD**: 両耳への伝搬遅延差。可変フラクショナル遅延で実装しており、移動時には自然なドップラー風の揺らぎも出る
- **ILD + 距離減衰**: 両耳それぞれへの 1/r ゲイン
- **頭部シャドウ**: 反対側の耳の高域減衰 (STFT ベースの時変フィルタ)
- **よそ見**: 高域カット + レベル低下。残響は減らないので「向こうを向いて喋っている」感が出る
- **距離感**: 合成 IR のルームリバーブ。遠いほど・よそ見ほど wet が増える

## ロードマップ / アイデア

- [x] 音源分離 (声だけ移動、BGM は静止) — `--separate`
- [ ] HRTF (SOFA ファイル) 畳み込みレンダラー。前後の定位が改善するはず
- [x] yt-dlp 連携 (URL を渡すと音声取得から一気に変換) — `yt`
- [ ] 移動パターンのプリセット (落ち着きめ / 動き多め など)
- [ ] プランのビジュアライズ (移動軌跡を上から見た図で確認)
- [ ] 足音・衣擦れなどの環境音の合成

## 開発

```sh
uv run pytest
```
