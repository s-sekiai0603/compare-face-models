# compare-face-models

顔識別モデルを比較検証するためのツールです。監視映像から顔画像を抽出し、顔特徴量モデル・顔ランドマークモデル・頭部姿勢モデルを画面で切り替えながら、顔の類似度比較レポート（Excel）を出力します。

抽出と顔ゲートは [premises-security](https://github.com/s-sekiai0603/premises-security) の顔識別（YOLOv8-Face → 骨格による顔ゲート → 適応マージンで切り出し）と同じです。特徴量マージ・類似度・追跡ID集約は、改修前の外見識別(reid)版と同じものです。

## 目次

- [機能と画面](#機能と画面)
- [使えるモデル](#使えるモデル)
- [動作環境とセットアップ](#動作環境とセットアップ)
- [モデル重みの配置（models/）](#モデル重みの配置models)
- [使い方（GUI）](#使い方gui)
- [コマンドライン](#コマンドライン)
- [精度の検証（eval）](#精度の検証eval)
- [入出力フォルダ構成](#入出力フォルダ構成)
- [処理の詳細](#処理の詳細)
- [Excel の構成](#excel-の構成)
- [モデルの追加方法](#モデルの追加方法)
- [ソース構成](#ソース構成)
- [実装メモ](#実装メモ)

## 機能と画面

| 画面 | 内容 |
|---|---|
| トップ | 「顔画像抽出」「レポート出力」を選択 |
| 顔画像抽出 | 映像一覧の表示、選択した映像から顔画像（と全身画像）を抽出 |
| 顔特徴量比較レポート出力 | 指定した顔画像または追跡IDと、比較対象フォルダ内の各追跡IDとの顔類似度を Excel レポートに出力。顔特徴量モデル・顔ランドマークモデル・頭部姿勢モデルを画面で切り替え |

GUI のほか、コマンドラインでも抽出・レポート出力ができます（「[コマンドライン](#コマンドライン)」参照）。公開ベンチマーク（CFP-FP 等）で各モデルの精度を論文の定義で測る `eval` コマンドもあります（「[精度の検証（eval）](#精度の検証eval)」参照）。

## 使えるモデル

3 種類のモデルは独立に選べます。レポート処理は「ランドマークモデルでアライン → 特徴量モデルで埋め込み → コサイン類似度 → 追跡ID集約」の順に流れます。

### 顔特徴量抽出モデル

| 画面の名前 | 内容 | 出典 | 重みの入手 |
|---|---|---|---|
| AdaFace | IR-101, WebFace12M 学習, 512 次元。公式コードどおりの前処理（BGR 入力、CLAHE なし） | [AdaFace (CVPR 2022)](https://github.com/mk-minchul/AdaFace) | 手動配置（公式の Pretrained Models から `adaface_ir101_webface12m.ckpt`） |
| CVLFace KP-RPE ViT-B (WebFace12M) | ViT-Base + KP-RPE + AdaFace, 512 次元。**5 点ランドマーク必須** | [CVLFace / KP-RPE (CVPR 2024)](https://github.com/mk-minchul/CVLface) | Hugging Face から自動取得（約 450MB） |
| LAFS Part fViT (WebFace4M) | ランドマーク予測 CNN 内蔵の ViT-Base。LAFS 自己教師あり事前学習 + 教師あり学習。1536 次元 | [LAFS (CVPR 2024)](https://github.com/szlbiubiubiu/LAFS_CVPR2024) | Google Drive から自動取得（約 900MB） |
| FSPFM ResNet50 (正面化あり, 微調整済) | ResNet50-IR + 特徴空間の姿勢正面化モジュール（HopeNet の姿勢埋め込みを入力）, 512 次元 | [FSPFM (ICIP 2025)](https://github.com/nikolaystanishev/feature-space-pose-frontalization) | Google Drive から自動取得（約 590MB） |

画面に出さない控え（CLI の `--model` では使えます。`list-models` に「非表示」と出ます）:
`FSPFM ResNet50 (正面化あり, 事前学習のみ)`（上の stage one、微調整前。Google Drive から自動取得、約 760MB）、
`CVLFace KP-RPE ViT-B (WebFace4M)`、`Part fViT (WebFace4M, 事前学習なし)`。

### 顔ランドマークモデル（特徴量抽出前のアライン）

| 画面の名前 | 内容 | 重みの入手 |
|---|---|---|
| なし | ランドマークを使わず 112x112 へリサイズするだけ。5 点座標は得られない | 不要 |
| InsightFace 2d106det | 106 点ランドマーク → 5 点 → ArcFace テンプレートへ相似変換（premises-security と同じ） | buffalo_l から自動取得 |
| InsightFace SCRFD (det_10g) | SCRFD で 5 点検出 → insightface の `norm_crop` と同じ相似変換。insightface の現行検出器 | buffalo_l から自動取得（約 290MB） |
| RetinaFace-R50 (Pytorch_Retinaface) | **論文どおり**: MS1M-RetinaFace（FSPFM）/ WebFace4M・12M（AdaFace, LAFS, CVLFace）の整列に使われた RetinaFace の PyTorch 再実装（ResNet-50, WIDER FACE 学習）。5 点検出 → `norm_crop` と同じ相似変換 | facexlib の GitHub Releases から自動取得（約 109MB） |
| MTCNN (AdaFace 公式) | **論文どおり**: AdaFace 公式デモ（`inference.py`）と同じ MTCNN（原著 Caffe 重み）と基準 5 点。MS1MV2 の整列手順と同じ | AdaFace リポジトリから自動取得（約 3MB） |
| CVLFace DFA | **論文どおり**: CVLFace 付属の Differentiable Face Aligner。KP-RPE の学習時と同じアライナ | Hugging Face から自動取得（約 2.5MB） |

- 「CVLFace KP-RPE」だけが 5 点座標を必要とします。座標が無い画像は ArcFace テンプレートの標準座標で代用し、件数を「条件」シートに記録します（「なし」を選ぶと全件が代用）。
- 各モデルの論文・学習データと同じ整列: AdaFace / LAFS / FSPFM は RetinaFace-R50（AdaFace 公式デモの手順を再現するなら MTCNN）、CVLFace KP-RPE は CVLFace DFA。
- 監視映像の小さい顔（15〜40px）では MTCNN の検出率が低く、SCRFD も精度が落ちます。その場合は RetinaFace-R50、2d106det、DFA が安定しています。「なし」は顔の向きが揃わないので非推奨です。

### 頭部姿勢モデル（情報欄に yaw / pitch / roll を表示）

| 画面の名前 | 内容 |
|---|---|
| なし | 表示しない |
| HopeNet (robust) | [HopeNet](https://github.com/natanielruiz/deep-head-pose) の `robust_alpha1`（実映像向けに公式が推奨）。Hugging Face ミラーから自動取得（約 96MB） |
| HopeNet (alpha2) | `alpha2`（FSPFM が内部で使う重み） |
| 6DRepNet | [6DRepNet](https://github.com/thohemp/6DRepNet)（ICIP 2022、MIT）。RepVGG-B1g2 + 6 次元回転表現。公式配布の `6DRepNet_300W_LP_AFLW2000.pth` を作者の OVGU クラウドから自動取得（約 150MB）。AFLW2000 の平均誤差 3.97 度（HopeNet 6.16 度）。学習時のクロップに合わせ、レポートでは全身画像と顔枠から余白の広い切り出しを作り直して入力する（全身画像が無ければ顔切り出しをそのまま使う）。PoC7 の低解像度映像では HopeNet (robust) より不安定だった（下記「頭部姿勢モデルの比較」） |

#### 頭部姿勢モデルの比較（PoC7 映像、2026-10-18）

本人 3 名・18 追跡IDの有効画像 234 枚（顔幅の中央値 約 25px）で HopeNet (robust) と 6DRepNet（余白広めクロップ）の yaw を比べた結果。

| 顔幅 | 枚数 | yaw 差の中央値 | 向き 5 区分の一致率 |
|---|---|---|---|
| 45px 以上 | 57 | 7° | 81% |
| 30〜45px | 48 | 9° | 62% |
| 20〜30px | 52 | 27° | 35% |
| 20px 未満 | 77 | 37° | 26% |

- 顔幅 45px 以上では両モデルがほぼ一致する。30px を切ると 6DRepNet は両目が見えている正面寄りの顔を −60〜−80°（右横顔）と出す傾向があり、
  RetinaFace の両目間隔が十分ある 149 枚を正面と判定した割合は HopeNet 50% に対し 6DRepNet 28%。真横の顔では左右の符号を取り違えることもある
  （20° 超の 128 枚中 22 枚で符号が逆）。
- 6DRepNet は AFLW2000 等のきれいな画像では HopeNet より高精度（平均誤差 3.97° 対 6.16°）だが、公開重みは画質劣化の増強無しで学習されており、
  監視映像の小さい・ぼけた顔には HopeNet の robust 重み（ぼけ・低解像度の増強あり）の方が安定。**当面の既定は HopeNet (robust)。**
  顔幅 45px 以上が確保できる映像なら 6DRepNet の方が角度の値は正確と考えられる。
- 向きの区分を 6DRepNet で付け直して角度別検証（顔識別の 1 位率）をやり直しても、全体の数字はほぼ変わらない（CVLFace 89% → 88%、AdaFace 84% → 84%）。
  区分が変わるのは主に小さい顔で、識別結果そのものは変わらないため。

類似度の計算には影響しません。横顔で各モデルの類似度がどう変わるかを確認する用途です。

### 固定のモデル（画面では選べない）

| 用途 | モデル |
|---|---|
| 人物検出・骨格 | YOLOv8l-pose（初回に自動取得・TensorRT 変換） |
| 追跡 | BoT-SORT（ultralytics 内蔵） |
| 顔検出 | YOLOv8s-Face（`models/detection/` のエンジン。読めなければ `.pt` を自動取得して変換） |

## 動作環境とセットアップ

- Windows / Python 3.10〜3.12 / [uv](https://docs.astral.sh/uv/)
- （GPU 利用時）NVIDIA GPU と CUDA 12.8 以上のドライバ（Blackwell 世代は必須）。CUDA Toolkit / TensorRT 本体の個別インストールは不要です（`uv sync` で pip パッケージとして入ります）。

```powershell
# uv のインストール（未導入の場合）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 依存関係のインストール（.venv が作成されます。初回は PyTorch 等で数分〜十数分）
uv sync

# 起動（どちらでも可）
uv run python -m compare_face
start.bat

# テスト（GPU・モデル重み不要。重みがあれば読込確認も走る）
uv run pytest
```

| 項目 | 設定 |
|---|---|
| PyTorch | CUDA 12.8 版（`tool.uv.sources` / `tool.uv.index` で指定） |
| TensorRT | `tensorrt-cu12==10.16.0.72` に固定。`models/detection/` の `.engine`（premises-security 側の環境で作成）はこの版製で、**エンジンは TensorRT の版が完全一致しないと読み込めない**ため同じ版にしています。版を変えると既存エンジンは使えなくなり、ツールは `.pt` からの再変換で動きます |
| ONNX Runtime | `onnxruntime`（CPU 版）。ランドマークモデル（ONNX）の推論に使用。GPU で動かす場合は `onnxruntime-gpu` に差し替え可 |
| CVLFace 用 | `huggingface_hub`（取得）、`omegaconf` / `timm` / `easydict` / `scikit-image`（取得したモデルコードが使用）。`transformers` は使いません |
| LAFS / FSPFM 用 | `gdown`（Google Drive からの取得） |

GPU がない環境でも `uv sync` は通り、CPU で動作します（低速）。

## モデル重みの配置（models/）

`models/` 配下は Git 管理外で、用途ごとのフォルダに分かれています。`python -m compare_face list-models` で状態を確認でき、`python -m compare_face download` で自動取得できるものをまとめて取得できます。

```
models/
  detection/   yolov8s-face-lindevs-b128.engine（顔検出、premises-security 由来）, yolov8s-face-lindevs.pt（予備）,
               yolov8l-pose.pt, yolov8l-pose-b16.engine（人物検出。自動取得・変換）
  landmark/    2d106det.onnx, det_10g.onnx（buffalo_l から自動取得）, retinaface/detection_Resnet50_Final.pth（facexlib）,
               mtcnn/pnet.npy, rnet.npy, onet.npy（AdaFace リポジトリ）, cvlface_DFA_mobilenet/（Hugging Face）
  feature/     adaface/adaface_ir101_webface12m.ckpt（手動配置）
               cvlface/cvlface_adaface_vit_base_kprpe_webface12m/（Hugging Face）
               lafs/lafs_partfvit_webface4m.pth（Google Drive）
               fspfm/fspfm_stage2.pt, fspfm_stage1.pt（Google Drive）
  pose/        hopenet/models/hopenet_robust_alpha1.pkl, hopenet_alpha2.pkl（Hugging Face ミラー）
               6drepnet/6DRepNet_300W_LP_AFLW2000.pth（公式配布）
```

- 手動配置が必要なのは AdaFace の重み（`adaface_ir101_webface12m.ckpt`。premises-security の `models/` と同じもの）と、顔検出エンジン（同 `yolov8s-face-lindevs-b128.engine`）だけです。エンジンが無い、または TensorRT の版が違って読めない場合は `yolov8s-face-lindevs.pt` を [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) から自動取得して `yolov8s-face-lindevs-b128-auto.engine` に変換します（既存のエンジンは上書きしません）。
- 他はレポート画面でモデルを選んだとき、または `download` コマンドで自動取得します（Hugging Face はトークン不要。Google Drive は制限で失敗することがあり、その場合は画面に出る手順で手動配置してください）。
- 各モデルのライセンス（コードは MIT 等、学習データの WebFace / MS1M は研究用途の条件あり）は業務利用の前に確認してください。
- **比較対象の特徴量キャッシュ**: レポート出力時、比較対象のアライン + 特徴量抽出の結果を比較対象フォルダ直下の `.embedding_cache/`（モデル × ランドマークごとに 1 ファイル）へ保存し、同じ比較対象フォルダでクエリだけ変えて出し直すときは再利用します（画像の追加・差し替えは自動で検出して計算し直し）。結果はキャッシュの有無で変わりません。「ログ」シートに再利用件数が出ます。使いたくない場合は CLI の `--no-cache`、または `.embedding_cache/` を削除してください。
- 精度検証用の検証セット（`cfp_fp.bin` 等）は `models/` ではなく `dataset/` に置きます（同じく Git 管理外。「[精度の検証（eval）](#精度の検証eval)」参照）。

## 使い方（GUI）

### 顔画像抽出

1. 起動して「顔画像抽出」を押す。
2. 「映像ルートフォルダ」「顔画像出力先」を選択する（前回の指定が初期表示されます）。
3. 一覧のチェックボックスで映像を選択（複数可）し、必要なら「抽出頻度(秒)」を変更して「抽出実行」を押す。追跡IDごとに、顔ゲートを通った顔が取れたフレームを基準に指定秒数間隔で出力します（既定 0.5 秒）。採用できる顔が無い間は、顔ゲート・品質ガード（採用基準）に外れた顔も間隔ごとに保存し、JSON に判定を記録します。使うかどうかはレポート時の「顔の採用基準」で決めます（既定は抽出時と同じ基準 = 採用された顔だけを使う）。
4. 初回は人物検出モデルの取得と TensorRT 変換で 1〜2 分かかります（ステータス欄に表示）。

### レポート出力

1. 「レポート出力」を押す。
2. クエリ指定方法を選ぶ。「顔画像指定」は `_face.jpg` 1 枚、「追跡ID指定」は追跡IDフォルダ内の全顔画像をクエリにします。
3. 比較対象フォルダ・レポート出力先、必要に応じて「全身画像も表示」「レポート出力件数」「特徴量マージ単位」「クエリ自動選別」「顔の採用基準」「探したい人物」を設定する。
4. 「顔特徴量抽出モデル」「顔ランドマークモデル」「頭部姿勢」を選び、「レポート作成開始」を押す。重みが無いモデルを選ぶと自動取得が始まり、取得できない場合は配置手順が表示されます。

### 検索元セット検索（総当たり・全体検索）

人が選んだ複数の顔画像（複数カメラ・複数追跡IDにまたがってよい）を 1 枚ずつ検索元にして、「セット内の総当たり」と「比較対象全体での本人の順位」を 1 つの Excel に出します。レポート出力が「1 人（1 枚または 1 追跡ID）× 全員」なのに対し、こちらは「N 枚 × N 枚」と「N 枚 × 全員」です。

1. 「検索元セット検索」を押す。検索元セットの置き場所は `顔画像出力先\検索元\` です（比較対象の走査ではこのフォルダを常に飛ばすので、比較対象フォルダの中にあっても二重に数えません）。
2. 「新規作成...」でセット名を入れると `顔画像出力先\検索元\<セット名>\` ができて Explorer で開きます。そこへ比較対象フォルダの顔画像（`_face.jpg`）をコピーします。サブフォルダ（例: `Cam4_79\01_正面.jpg`）で分けると、Excel の「グループ」になります。ファイル名は変えてもかまいません。
3. 「更新」で画像数を確認し、比較対象フォルダ・出力先・モデルを選んで「検索開始」を押す。

- **元画像の特定**: 検索元の各画像について、比較対象フォルダ内の同じ内容の画像（バイト一致。old フォルダも探す）を元画像とし、カメラID・追跡ID・顔信頼度・採用基準などを引き継ぎます。元画像の追跡IDは自動で「本人」になります。元画像が無い画像（縮小版など）も検索元には使えますが、カメラ等は「-」になり、本人の追跡IDには数えません。
- **本人の追跡ID（追加）**: 検索元に入れていない本人の追跡ID（2 回目の通過など）は「カメラID|映像ファイル名|追跡ID」を 1 行 1 件で書きます（CLI は `--person`）。
- **比較対象の扱い**: レポートと同じく、無効画像・old フォルダ（「old フォルダの画像も含める」で戻せる）・顔の採用基準で外れた画像は比較対象から外します。検索元自身の画像 1 枚だけを比較対象から外して順位を数えます。検索元の元画像が比較対象に無い（old・無効・採用基準外）場合、その画像を検索対象にした順位は出ません（「検索元」シートの「比較対象での扱い」に理由）。
- **出力** `query_set_<日時>.xlsx`: 条件 / 検索元（各画像の元画像・顔信頼度・頭部姿勢・比較対象での扱い）/ 総当たり（表）（N×N の類似度に色付け）/ 総当たり（一覧）（N×(N−1) 組を 1 行ずつ。同じグループに○）/ 全体検索（画像）（検索元ごとに、セット内の他の画像とその追跡IDが比較対象全体で何位か。全体・カメラ別・他人だけ）/ 全体検索（追跡ID）（検索元 × 本人の追跡IDの順位と、他人だけで 1 位・10 位以内・100 位以内の集計）/ ログ。順位の色は緑 10 位以内・黄 100 位以内・橙 それ以上です。

## コマンドライン

```powershell
# 登録済みモデルと重みの状態
uv run python -m compare_face list-models

# 自動取得に対応したモデルの重みをまとめて取得（--model / --landmark / --pose で絞り込み可）
uv run python -m compare_face download

# 顔画像抽出（--video で映像ファイル名を絞り込み可。省略時はルート配下の全映像）
uv run python -m compare_face extract --video-root D:\videos --out D:\faces --interval 0.5 --video 20260311122723.mp4

# レポート出力（--query が _face.jpg なら顔画像指定、フォルダなら追跡ID指定）
uv run python -m compare_face report --query D:\faces\...\3\20260311_122723_500_face.jpg --gallery D:\faces --out D:\reports `
    --model AdaFace --landmark "InsightFace SCRFD (det_10g)" --limit 500 --target "Cam1|20260311122723.mp4|3"
uv run python -m compare_face report --query D:\faces\...\3 --gallery D:\faces --out D:\reports `
    --model "CVLFace KP-RPE ViT-B (WebFace12M)" --landmark "CVLFace DFA" --merge-sec 1.5 --no-body
uv run python -m compare_face report --query D:\faces\...\3 --gallery D:\faces --out D:\reports `
    --model "FSPFM ResNet50 (正面化あり, 微調整済)" --landmark "InsightFace SCRFD (det_10g)" --pose "HopeNet (robust)"
# old フォルダに移した画像も含めて（選別前の状態で）出す / 特徴量キャッシュを使わない
uv run python -m compare_face report --query D:\faces\...\3 --gallery D:\faces --out D:\reports --include-old --no-cache
# クエリを自動選別（顔信頼度 0.65 / 品質ノルム 17 / 顔幅 30px 未満の画像を使わない）
uv run python -m compare_face report --query D:\faces\...\3 --gallery D:\faces --out D:\reports `
    --min-face-conf 0.65 --min-norm 17 --min-face-width 30
# クエリ画像を別フォルダに集めて使うとき、元の追跡IDを比較対象から外す（複数指定可。「条件」シートに記録される）
uv run python -m compare_face report --query D:\queries\右横顔 --gallery D:\faces --out D:\reports --exclude-dir D:\faces\...\3
# 顔の採用基準を緩める（横顔も使う: 両目間隔比の判定なし / 全て使う: --accept-all）。既定は抽出時と同じ基準
uv run python -m compare_face report --query D:\faces\...\3 --gallery D:\faces --out D:\reports --min-eye-ratio 0
uv run python -m compare_face report --query D:\faces\...\3 --gallery D:\faces --out D:\reports --accept-all
# 検索元セット検索（D:\faces\検索元\Aさん にコピーした顔画像の総当たり・全体検索。--person で本人の追跡IDを追加）
uv run python -m compare_face query-set --set D:\faces\検索元\Aさん --gallery D:\faces --out D:\reports `
    --model "CVLFace KP-RPE ViT-B (WebFace12M)" --landmark "RetinaFace-R50 (Pytorch_Retinaface)" --pose "HopeNet (robust)" `
    --person "Cam1|20260311113522.mp4|624"
# 抽出済みの顔画像に「顔枠に本人の鼻点が無い」無効判定を後から適用
uv run python -m compare_face revalidate --faces D:\faces
```

公開ベンチマークでの精度測定は `eval` コマンドで行います（次節）。

終了コードは成功 0、実行時エラー 1、引数エラー 2 です。モデルを変えて同じ条件でレポートを出す、といった自動検証はこのコマンドで行えます。`start.bat extract ...` のように `start.bat` に引数を渡しても同じです。

## 精度の検証（eval）

insightface 形式の検証セット（`lfw.bin` / `cfp_fp.bin` / `cplfw.bin` / `calfw.bin` / `agedb_30.bin`。112x112 にアライン済みの顔ペアと同一人物フラグ）で、各モデルの 1:1 照合精度を論文と同じ定義で測り、公表値と並べます。検証セットは単体配布されておらず、[insightface Dataset Zoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) の学習データパッケージ（CASIA-WebFace が最小、MS1M-ArcFace 等）に同梱されています（研究・非商用の配布条件あり）。`.bin` を `dataset/`（Git 管理外）に置くと `eval` コマンドが自動で見つけます。

```powershell
# dataset/ 配下の全 .bin × 画面に出る全モデル × ランドマーク「なし」。結果は dataset/eval_<日時>.csv
uv run python -m compare_face eval

# 対象を絞る / ランドマークモデルでアラインし直して比べる / 等間隔に選んだ 500 ペアで動作確認
uv run python -m compare_face eval --bin dataset\cfp_fp.bin --model AdaFace --model "CVLFace KP-RPE ViT-B (WebFace12M)" `
    --landmark なし --landmark "RetinaFace-R50 (Pytorch_Retinaface)" --out D:\eval.csv --limit 500
```

- **精度の定義**: 特徴量を L2 正規化し、ペアの二乗ユークリッド距離（= 2 − 2cos）に対するしきい値を 10 分割交差検証で決めた正解率の平均 ± 標準偏差。insightface の `verification.py` と同じ手順（しきい値 0〜4 を 0.01 刻み、分割はシャッフルなし、9 分割で最良しきい値を選び残り 1 分割で測る）。
- **flip 融合あり / なし**: 「あり」は元画像と左右反転画像の特徴量を足して正規化したもの（insightface / AdaFace の flip test。公表値と同じ条件）。「なし」はレポート出力と同じ条件（モデル内部の TTA のみ）。KP-RPE には反転時に鏡像化した 5 点を渡します。
- **ランドマークモデル**: `.bin` は整列済みなので通常は「なし」で評価します。他を指定すると整列済み画像を検出・アラインし直してから特徴量化するので、検出器の違いが精度に与える影響を測れます（検出できない画像はリサイズで代替し「アライン失敗」に件数が出ます）。
- **整列失敗ペア除外**: 検証セット側の整列失敗（横顔のランドマーク誤検出で肌や背景だけになった画像。顔の中央領域の標準偏差 < 15 で判定）を数え、それを含むペアを除いた精度も出します。どのモデルでも照合できないペアなので、公表値との差がモデル側かデータ側かを切り分けるための値です。`.bin` の版によって件数が違います（手元の `cfp_fp.bin` では 14000 枚中 200 枚、7000 ペア中 200 ペア）。
- **CSV の列**: データセット, 顔特徴量モデル, 顔ランドマークモデル, ペア数, 精度 flip融合あり(%), 標準偏差(%), 精度 flip融合なし(%), 精度 整列失敗ペア除外(%), 整列失敗画像(枚), 整列失敗ペア, しきい値(cos), アライン失敗, ランドマーク未取得, 公表値(%), 処理時間(秒)。1 行終わるごとに保存するので途中で止めても結果が残ります。公表値と出典は `core/eval/published.py` にあります。
- **非表示モデル**も `--model` で指定できます（`FSPFM ResNet50 (正面化あり, 事前学習のみ)` 等）。

### CFP-FP での結果

2026-10-09 時点、手元の `cfp_fp.bin`（7000 ペア / 14000 枚、insightface 配布）での実測値です。精度は flip 融合あり・10 分割交差検証（平均 ± 標準偏差）。「除外後」は整列失敗ペア 200 件を除いた 6800 ペアでの精度、「公表値」は論文・公式 README の値です。

| 顔特徴量モデル（ランドマーク「なし」） | 精度 (%) | flip 融合なし (%) | 除外後 (%) | 公表値 (%) |
|---|---|---|---|---|
| LAFS Part fViT (WebFace4M) | 97.76 ± 0.92 | 97.69 | 99.18 | 99.18 |
| AdaFace | 97.66 ± 1.02 | 97.66 | 99.10 | 99.26 |
| Part fViT (WebFace4M, 事前学習なし)（非表示） | 97.64 ± 0.84 | 97.73 | 98.91 | 99.13 |
| CVLFace KP-RPE ViT-B (WebFace12M)（5 点はテンプレート座標） | 97.40 ± 0.99 | 97.36 | 98.84 | 99.30 |
| FSPFM ResNet50 (正面化あり, 事前学習のみ)（非表示） | 97.03 ± 0.88 | 96.73 | 98.46 | 98.60 |
| CVLFace KP-RPE ViT-B (WebFace4M)（非表示、テンプレート座標） | 96.97 ± 1.02 | 97.03 | 98.37 | 99.01 |
| FSPFM ResNet50 (正面化あり, 微調整済) | 96.87 ± 0.99 | 96.70 | 98.31 | 98.67 |

ランドマークモデルで整列済み画像を検出・アラインし直した場合（AdaFace と CVLFace KP-RPE）:

| 顔ランドマークモデル | AdaFace 精度 (%) | AdaFace 除外後 (%) | CVLFace 精度 (%) | CVLFace 除外後 (%) | アライン失敗（枚） |
|---|---|---|---|---|---|
| なし（リサイズのみ） | 97.66 | 99.10 | 97.40 | 98.84 | 0 |
| InsightFace 2d106det | 97.47 | 98.91 | 97.61 | 99.06 | 0 |
| InsightFace SCRFD (det_10g) | 97.87 | 99.32 | 97.87 | 99.32 | 208 |
| RetinaFace-R50 (Pytorch_Retinaface) | **97.94** | **99.40** | **97.96** | **99.41** | 217 |
| MTCNN (AdaFace 公式) | 97.59 | 99.03 | 97.54 | 98.99 | 260 |
| CVLFace DFA | 97.80 | 99.25 | 97.87 | 99.32 | 4 |

読み方:

- **公表値との差はほぼ検証セット側の整列失敗**です。手元の `cfp_fp.bin` には顔の中央が一色（肌・背景・耳だけ）の画像が 200 枚（重複を除くと 53 枚）あり、それを含む 200 ペア（うち同一人物 105 ペア）はどのモデルでも照合できません。このペアを除くと、LAFS は公表値と一致（99.18）、AdaFace は −0.16、FSPFM は −0.14〜−0.36、CVLFace は DFA / RetinaFace の 5 点を使えば公表値と同等（99.32〜99.41 vs 99.30）になります。実装（前処理・ネットワーク・重み・評価手順）は公表値を再現できる状態です。
- insightface の `cfp_fp.bin` は配布時期で整列が異なります（[issue #467](https://github.com/deepinsight/insightface/issues/467)）。MS1M-ArcFace（faces_emore）または MS1M-RetinaFace パッケージの `cfp_fp.bin` で同じコマンドを実行すると、公表値と直接比較できる数値になるはずです。
- **ランドマークモデル**: 整列済み画像を RetinaFace-R50 または SCRFD で整列し直すと 0.2〜0.3 ポイント上がります（学習データの整列器が RetinaFace のため）。CVLFace KP-RPE は 5 点座標をテンプレートで代用すると 0.5 ポイント下がるので、画面では DFA か RetinaFace-R50 を選んでください。MTCNN と 2d106det は再整列しても上がりません。アライン失敗の大半（208〜217 枚）は上記の整列失敗画像で、DFA だけはそれらにも 5 点を返します（信頼度を閾値にしていないため）。
- **モデル間の差**: 「なし」での上位 4 モデルの差は 0.4 ポイント以内で、10 分割の標準偏差（約 0.9）より小さく有意ではありません。CFP-FP（正面と横顔の照合）では LAFS / AdaFace / CVLFace（DFA か RetinaFace 併用）が同等、FSPFM は約 0.7 ポイント下です。
- flip 融合（公表値の条件）とレポートと同じ条件（融合なし）の差は 0.1 ポイント以内です。

## 入出力フォルダ構成

### 入力

```
ルートフォルダ\(ロケーション)\(カメラID)\(mp4形式映像ファイル)
```

ファイル名は `YYYYMMDDHH24MISS.mp4`（例: `20260311122723.mp4` → 2026/03/11 12:27:23）を想定します。日本語を含むフォルダ名・ファイル名に対応しています。

### 出力

```
出力先ルートフォルダ\(YYYYMMDDHH24MISS)\(カメラID)\(映像ファイル名・拡張子なし)\(追跡ID)\
    {YYYYMMDD_HHMMSS_mmm}_face.jpg   ... 顔の切り出し（適応マージン拡張後。リサイズなし）
    {YYYYMMDD_HHMMSS_mmm}_body.jpg   ... 人物bboxの切り出し（表示用）
    {追跡ID}.json                    ... 顔・骨格・信頼度
```

JSON は追跡IDフォルダごとに 1 ファイルで、画像ごとに次の項目を持ちます。

| 項目 | 内容 |
|---|---|
| `file_face` / `file_body` | 画像ファイル名 |
| `frame_index` / `time_sec` / `timestamp` | フレーム番号、映像内経過秒、撮影日時（ファイル名の日時 + 経過時間） |
| `bbox_xyxy` / `person_conf` | 人物の検出 box（フレーム座標）と信頼度（YOLOv8-pose） |
| `face_bbox_xyxy` / `face_conf` | 顔の検出 box（フレーム座標）と信頼度（YOLOv8-Face） |
| `face_crop_xyxy` | 適応マージンで拡張した切り出し範囲（`_face.jpg` の範囲） |
| `face_source` | `frame`=フレーム全体の顔検出 / `body_crop`=全身クロップでの再検出 |
| `face_width` / `face_height` | 検出顔 box のサイズ(px) |
| `foot_xy` / `foot_source` | 足元の点 `[x, y]`（フレーム座標）。左右の足首キーポイント（信頼度 0.15 以上）の平均で `foot_source` は `ankle`。取れなければ全身 box の下辺中央で `bbox_bottom` |
| `gate_ok` / `gate_reason` | 採用基準（顔ゲート + 品質ガード）を通ったか（= premises-security が出力する顔か）と、外れた理由 |
| `face_visible` | 顔ゲートの骨格ルール（鼻+片目が見える / 下向きでない）を通ったか |
| `crop_min_side` | 切り出し box の短辺(px)。品質ガード（60px）の判定値（小顔拡大後の座標） |
| `eye_ratio` / `frontal` | 両目間隔÷顔幅（正面度）と、0.25 以上で正面とみなしたフラグ。片目しか見えない場合は `null` / `false` |
| `nose_conf` / `l_eye_conf` / `r_eye_conf` | 骨格キーポイントの信頼度（顔ゲートの判定に使用） |
| `quality` | 品質スコア = 顔信頼度 + min(10, 顔幅/10) |
| `invalid` / `invalid_reason` | 無効画像（別人の骨格を含む）かどうか |
| `keypoints` | YOLOv8-pose の 17 点（COCO 順）の座標と信頼度 |

## 処理の詳細

### 抽出（`core/extract/`）

| 段 | 内容 |
|---|---|
| デコード先読み | 映像デコード専用スレッドが 64 フレームまで先読み |
| バッチ推論 | 16 フレームずつ、人物（YOLOv8-pose, conf 0.25）と顔（YOLOv8-Face, conf 0.30 / IoU 0.60 / 640px）を推論 |
| 追跡 | 人物検出を**フレーム順に 1 枚ずつ** BoT-SORT へ渡す（追跡 ID 付与）。全身向けのハードフィルタ（足首必須）は適用しない |
| 顔と人物の紐付け | 顔 box の中心が人物 box 内にある顔をその人物の顔とする（複数の人物に含まれる場合は面積最小の人物、1 人物に複数の顔は信頼度最大） |
| 顔の再検出 | 出力対象なのに顔が付かなかった人物は、全身クロップに対して YOLOv8-Face を再実行し、遠くの小さい顔を拾う |
| 顔ゲート | 骨格キーポイント（信頼度 0.15 以上）で判定: 鼻+片目以上が見えている / 耳が目より上にある下向き顔でない / 両目が見える場合は 両目間隔÷顔幅 ≥ 0.05。結果は JSON（`face_visible` / `eye_ratio` / `gate_reason`）に記録し、外れた顔も保存する |
| 適応マージン | 顔幅 < 40px: 30% / < 80px: 20% / それ以上: 10% だけ顔 box を全方向に拡張 |
| 小顔前処理 | 拡張後の短辺 < 96px なら、周辺 ROI を長辺 800px に拡大してから切り出す |
| 品質ガード | 切り出し box の短辺 < 60px（小顔拡大後の座標で判定。premises-security の `AdaFaceEmbedder._crop_with_margin` はこれを出力しない）。短辺を JSON の `crop_min_side` に記録し、外れた顔も保存する |
| 採用基準 | 顔ゲートと品質ガードの両方を通った顔が `gate_ok=true`（premises-security が出力する顔）。基準外の顔は、採用できる顔が間隔内に無いときだけ間隔ごとに保存する（採用された顔の直後は出さない）ので、採用される顔の出力タイミングと処理量は基準外の顔を保存しない場合と同じ |
| 無効判定 | 切り出し範囲に別人の骨格点（信頼度 0.3 以上）が 3 点以上入る画像は `invalid` にする |
| 保存 | `_face.jpg`（リサイズなし）、`_body.jpg`、JSON。JPEG 品質 90 |

premises-security との違いは 3 点です。(1) premises-security は全身画像を間隔出力しその瞬間に顔があれば顔も出しますが、本ツールは顔が出せたフレームを基準に間隔を数えます。(2) premises-security は切り出し後に 112x112 へリサイズして保存しますが、本ツールはリサイズ前の切り出しを保存し、レポート時に各モデルの入力サイズへ整えます。(3) premises-security は顔ゲート・品質ガードに外れた顔を捨てますが、本ツールは判定を記録して保存し、使うかどうかをレポート時の「顔の採用基準」で決めます（既定は同じ基準なので結果は変わりません）。

### レポート（`core/report/`、`core/landmarks/`、`core/features/`、`core/pose/`）

1. **アライン**: ランドマークモデルで `_face.jpg` を 112x112 に整え、5 点座標（左目・右目・鼻先・左口角・右口角）があれば特徴量モデルへ渡す。失敗した画像は単純リサイズで代替し、件数を「条件」シートに記録。
   - なし: INTER_CUBIC でリサイズ。
   - InsightFace 2d106det: 192x192 で 106 点を推定 → 5 点に集約 → ArcFace テンプレートへ相似変換。5 点が切り出し範囲の外（10% マージン超）なら失敗。
   - InsightFace SCRFD: 切り出し画像を 640px キャンバスの中央に半分の大きさで置き（顔が全面を占める画像をそのまま広げると検出できないため）、SCRFD（det_10g）で検出した最大の顔の 5 点を insightface `norm_crop` と同じ相似変換（skimage の最小二乗）でアライン。しきい値 0.5 で見つからなければ 0.05 で再試行（切り出しは顔と確定しているため）。それでも検出なしなら失敗。
   - RetinaFace-R50: 切り出しの周囲に各辺 50% のグレー余白を足し（長辺 1024px 上限）、元コード `detect.py` と同じ手順（BGR 平均減算、prior box、NMS 0.4）で検出。最もスコアの高い顔の 5 点を `norm_crop` と同じ相似変換でアライン。しきい値 0.6 で見つからなければ 0.05 で再試行。
   - MTCNN: 切り出しの周囲に各辺 25% の黒余白を足し、AdaFace 公式と同じ既定値（最小顔 20px、しきい値 0.6/0.7/0.9、縮小率 0.85）で P/R/O-Net を実行し、最もスコアの高い顔の 5 点を基準 5 点（= ArcFace テンプレート）へ相似変換。公式設定で見つからなければ緩いしきい値で再試行し、それでも検出なしなら失敗。
   - CVLFace DFA: RetinaFace-MobileNet で 5 点検出 → ArcFace テンプレートへ相似変換（112x112 出力）。検出スコアは閾値にせず、5 点の配置が顔として妥当かだけで失敗判定。
2. **特徴量抽出**（FP32、バッチ推論）:
   - AdaFace: BGR のまま → (x/255−0.5)/0.5 → IR-101 → 左右反転 TTA 平均 → L2 正規化（512 次元）。チャネル順は公式 `inference.py`（RGB を `[:, :, ::-1]` で BGR にして入力）と学習データ変換 `convert.py`（検証コードが cv2 の BGR なので意図的に入れ替えて保存）に合わせています。CFP-FP では BGR 97.66% / RGB 97.47% と差は小さいです。他の 3 モデル（CVLFace / LAFS / FSPFM）は RGB が正しく、CFP-FP で BGR にすると 0.2〜1.3 ポイント下がることを確認しています。
   - CVLFace KP-RPE: RGB → (x/255−0.5)/0.5 → ViT-Base + KP-RPE（5 点を [0,1] 正規化して入力）→ 左右反転 TTA 平均（反転時は座標も鏡像化し左右の目・口角を入れ替え）→ L2 正規化（512 次元）。CUDA 拡張 `rpe_ops` はビルドせず純 PyTorch 実装で推論。
   - LAFS Part fViT: RGB → x/255−0.5（[-0.5, 0.5]）→ Part fViT（内蔵 MobileNetV3 が 196 点を予測し各点周辺の 8x8 パッチをトークン化）→ 元画像と左右反転の特徴（各 768 次元）を**連結** → L2 正規化（1536 次元）。元リポジトリの IJB 評価と同じ flip test。
   - FSPFM: RGB → (x/255−0.5)/0.5 → ResNet50-IR（512 次元）→ 特徴空間の姿勢正面化モジュール → L2 正規化。姿勢は同じアライン済み画像を 224x224 にして HopeNet（alpha2）に入れた**分類層手前の 2048 次元埋め込み**（元コードどおり。オイラー角そのものではない）。TTA は元リポジトリの評価既定に合わせて無し。
3. **類似度**: コサイン類似度。同値（小数 4 桁）は同順位の標準競技順位。画像指定はクエリ 1 枚 × 比較対象全画像、追跡ID指定はクエリ全画像 × 比較対象全画像。
4. **追跡ID集約**: 追跡IDごとに最大類似度の 1 件を残して再順位付け（max 集約）。
5. **頭部姿勢（任意）**: HopeNet または 6DRepNet で各顔クロップから yaw / pitch / roll を推定し情報欄に表示（HopeNet は `_face.jpg` をそのまま 224x224 に、6DRepNet は全身画像から余白広めに切り出し直して入力）。

バッチ枚数は結果に影響しません（モデルは eval 状態で画像ごとに独立）。VRAM が足りない場合は各 `features/*.py` の `EMBED_BATCH` を下げてください。

### 特徴量マージ（追跡ID指定時）

「特徴量マージ単位」に秒数を指定すると、クエリ画像を時刻順に**指定秒数ごとの固定区間**でグループ化し（画像の無い区間は飛ばして G1, G2, … と採番）、グループ内の特徴量を平均して L2 正規化したものをクエリにします。レポートではクエリ画像に「クエリグループID:Gn」、ギャラリー画像に「ヒットしたクエリグループID:Gn」を表示します。

### 画像の手動除外（old フォルダ）

追跡IDフォルダの中に `old` フォルダを作り、使いたくない顔画像（`_face.jpg` と対になる `_body.jpg`）を移すと、その画像はクエリ・比較対象の両方から外れます（JSON は変更不要。元に戻すにはファイルを戻すだけ）。別人の顔が混ざった画像や、ぼやけた画像を人の判断で外すときに使います。レポートの「クエリ画像数」「比較対象 画像数」は残った画像だけを数えます。

選別の効果を比べたいときは、レポート画面の「old フォルダの画像も含める」（CLI は `--include-old`）を有効にすると、ファイルを動かさずに `old` の画像を元の追跡IDの画像として戻した状態（選別前）で比較できます。どちらで出したかは「条件」シートの「old フォルダの画像」の行に記録されます。

### 顔の採用基準（抽出時の顔ゲート・品質ガードの再判定）

抽出は顔ゲート・品質ガードに外れた顔も保存するので、どこまで使うかをレポート時に決めます。レポート画面の「顔の採用基準」（CLI は `--no-face-gate` / `--min-eye-ratio` / `--min-crop-side` / `--accept-all`）で、クエリ・比較対象の両方に適用します（「条件」シートに記録）。

| 項目 | 意味 | 既定（抽出時と同じ） |
|---|---|---|
| 骨格ルール | 鼻+片目が見えている / 耳が目より上にある下向き顔でない（JSON の `face_visible`） | 適用する |
| 両目間隔比 | 両目間隔÷顔幅（JSON の `eye_ratio`）。両目が見える画像だけ判定。横顔を外す | 0.05（0 で判定しない） |
| 切り出し短辺 | 切り出し box の短辺(px)（JSON の `crop_min_side`）。小さい顔を外す | 60px（0 で判定しない） |

既定のままなら抽出時に採用された顔（`gate_ok=true`）だけを使い、従来と同じ結果になります。緩めると抽出時に不採用だった横顔・小さい顔も比較に使えます（情報欄に「抽出時は採用基準外:理由」を表示）。厳しくすると採用された顔をさらに絞れます。外した画像は、比較対象からは除外し、クエリでは薄い青の背景と「【採用基準外】特徴量比較に未使用」を表示します。判定値の無い古い抽出結果（この機能を入れる前の出力）は採用された顔しか保存していないので、基準を通ったものとして扱います。

### クエリの自動選別と品質スコア

CLI の `--clahe` を付けると、アライン後の顔画像に CLAHE（LAB の L チャネル, clip 2.0。premises-security の AdaFace 前処理と同じ）を掛けてから特徴量化します（クエリ・比較対象とも。比較対象の特徴量キャッシュは `_clahe` 付きの別ファイル。「条件」シートに記録）。

レポート画面の「クエリ自動選別」（CLI は `--min-face-conf` / `--min-norm` / `--min-face-width`）にしきい値を入れると、クエリ画像のうち条件に外れるものを特徴量比較に使いません（レポートでは薄い橙の背景と理由を表示。「条件」シートにしきい値と除外数を記録）。0 はその条件を使いません。

| しきい値 | 意味 | 目安 |
|---|---|---|
| 顔信頼度 | 顔検出（YOLOv8-Face）の信頼度。ぼけ・横顔で下がる | 0.65 |
| 品質ノルム | AdaFace IR-101 の正規化前の特徴量ノルム。AdaFace 損失で学習したモデルは画質が低いほど小さい。識別モデルに関係なく常に AdaFace で測る（AdaFace の重みが必要） | 17（監視映像の下位 13%。横顔は低く出るので 20 以上にはしない） |
| 顔幅 | 検出した顔枠の幅(px)。ノルム・信頼度は顔の大きさと無関係なので、小さい顔は幅で外す | 30px |

比較対象には適用しません（監視映像では顔の半分以上がこの水準を下回り、しきい値をかけると追跡ID（人物）の 2〜6 割が検索不能になるため）。品質ノルムは計算したクエリの情報欄に「ノルム:xx」として表示します。

### 無効画像

次の画像は無効とみなし、JSON の `invalid` / `invalid_reason` に記録します。

1. 切り出し範囲内に別の人物の骨格点（信頼度 0.3 以上）が 3 点以上含まれる（別人の骨格）。
2. 本人の鼻のキーポイント（信頼度 0.15 以上）が顔枠から顔幅の半分より離れている（顔検出枠が隣の人物の顔や後頭部に付いた画像。許容を狭くすると小さい顔のキーポイント誤差まで拾うため半分にしている）。抽出済みの出力には `python -m compare_face revalidate --faces <顔画像出力先>` で後から適用できます（他の理由で無効の画像は変更せず、このルールで無効になった画像は判定し直します）。無効画像は特徴量抽出・類似度計算に使わず、比較対象からも除外します。クエリ画像としてはレポートに出力し、情報欄に「【無効画像】特徴量比較に未使用」と無効理由を表示、背景をグレーにします。クエリが全て無効の場合はエラーにします。

## Excel の構成

| シート | 内容 |
|---|---|
| 条件 | 生成日時、顔特徴量抽出モデル、顔ランドマークモデル、頭部姿勢モデル、クエリ/比較対象/出力先、全身画像の表示、old フォルダの画像、顔の採用基準、クエリ自動選別、各件数（無効・採用基準外・自動選別の除外数を含む）、アライン失敗数、ランドマーク未取得数 |
| 探したい人物検索結果 | 「探したい人物」を指定したときだけ作成。指定した各人物が「顔比較(追跡ID集約)」の何位に出たか（順位・類似度・ロケーション・カメラID・追跡ID・映像・撮影日時・顔信頼度・ヒットしたクエリ・顔画像・全身画像）。上位件数に入らなかった場合・比較対象に無い場合・クエリ自身（比較対象から除外）や `--exclude-dir` で除外した追跡IDの場合はその旨を表示。改修前の外見識別(reid)版と同じ形式 |
| 顔比較 | クエリ画像の下に、類似度降順（上位N件、既定500）を 10 列グリッドで表示 |
| 顔比較(追跡ID集約) | 追跡IDごとに最大類似度の 1 件 |
| ログ | レポート出力開始〜完了までの各フェーズの開始/終了時刻・所要時間・備考 |

各セルは「情報行 + 顔画像行（+ 全身画像行）」で構成します。情報行には順位・類似度・ロケーション・カメラID・追跡ID・人物信頼度・顔信頼度・顔幅・切り出し短辺・正面度・品質・（頭部姿勢を選んだ場合）yaw/pitch/roll・映像ファイル・撮影日時を表示します。「探したい人物」に該当する画像はピンク、クエリは緑、無効画像はグレー、採用基準外のクエリは薄い青、自動選別で除外したクエリは薄い橙です。

「探したい人物」欄には、カメラID・映像ファイル名・追跡ID の組を最大 10 件指定できます。入力のある項目だけで照合し（空欄は不問）、映像ファイル名は拡張子の有無を問いません。カメラID欄に「カメラID|映像ファイル名|追跡ID」のように `|` 区切りで貼り付けると各欄に振り分けます。指定内容は「条件」シートに記録され、次回画面を開いたときに復元されます。

## モデルの追加方法

### 顔特徴量抽出モデル

`core/features/` に `FaceFeatureExtractor`（`core/features/base.py`）を継承したクラスを作り、`@register` を付けて `core/features/__init__.py` で import します。レジストリに登録されたモデルがレポート画面のラジオボタンと `list-models` に自動で並びます。

```python
from .base import FaceFeatureExtractor, register

@register
class MyFaceModel(FaceFeatureExtractor):
    name = "MyModel"          # 画面・レポートに表示する名前（settings.json / --model にも使う）
    description = "説明"       # ツールチップ
    input_size = 112          # アライン後の入力サイズ（ランドマークモデルがこのサイズに整える）
    embed_dim = 512
    needs_landmarks = False   # True にすると embed_batch に 5 点ランドマーク（画像ごとの (5,2) ピクセル座標 or None）が渡される
    hidden = False            # True にすると画面の選択肢に出さない（CLI の --model では使える）

    def weights_ready(self) -> bool: ...        # 重みがあるか（置き場所は core/common/weights.py の FEATURE_DIR 配下）
    def manual_instructions(self) -> str: ...   # 無いときに画面に出す配置手順
    def download_weights(self, log) -> None: ...  # 任意。自動取得できるモデルは実装する（既定は配置手順の通知）
    def load(self) -> None: ...                 # モデル読込（実行環境を print する）
    def embed_batch(self, faces_bgr, landmarks=None) -> np.ndarray: ...  # (N, embed_dim) の L2 正規化済み特徴量
```

- AdaFace の別アーキテクチャは `features/adaface.py` の `_AdaFaceBase`、CVLFace の別モデルは `features/cvlface.py` の `_CVLFaceBase`、LAFS の別チェックポイントは `features/lafs.py` の `_LAFSBase`、FSPFM の別チェックポイントは `features/fspfm.py` の `_FSPFMBase` を継承すれば、重みファイル名（と取得先）の定義だけで追加できます。
- 配布コードをそのまま使えない場合（学習用の依存を含む等）は、推論経路だけを `features/*_net.py` に移植し、配布チェックポイントの全キーが一致することをテストで確認してください（LAFS / FSPFM がその例）。

### 顔ランドマークモデル

`core/landmarks/` に `FaceAligner`（`core/landmarks/base.py`）を継承したクラスを作り、`@register` を付けて `core/landmarks/__init__.py` で import します。`align(face_bgr, out_size)` で保存済みの顔クロップを `out_size` 正方形のアライン済み画像にして返します（失敗時は `None`）。5 点座標を持つモデルは `align_with_landmarks` を実装し、`AlignResult(image, landmarks)` で**アライン後画像内の 5 点ピクセル座標**（左目・右目・鼻先・左口角・右口角の順）も返します。

### 頭部姿勢モデル

`core/pose/__init__.py` の `POSE_OPTIONS` に「名前 -> 推定器を作る関数」を追加します。別のネットワークを使う場合は `HeadPoseEstimator`（HopeNet）と同じインターフェース（`name` / `weights_path` / `weights_ready` / `manual_instructions` / `download_weights` / `load` / `angles`）のクラスを作ります（例: `core/pose/sixdrepnet.py` の `SixDRepNetEstimator`）。

## ソース構成

```
src/compare_face/
  cli.py, app.py, ui/      コマンドラインと GUI（PySide6）。処理本体は持たない
  core/common/             共通部品（重みの置き場所、設定、日本語パス対応の画像入出力、ログ、ArcFace テンプレート、Hugging Face リポジトリ読込）
  core/extract/            抽出パイプライン（映像走査、人物検出・追跡、顔検出、顔ゲート・クロップ、無効判定）
  core/landmarks/          顔ランドマークモデルのレジストリ
  core/features/           顔特徴量モデルのレジストリ（*_net.py は移植したネットワーク定義）
  core/pose/               頭部姿勢のレジストリ（HopeNet / 6DRepNet）
  core/report/             レポートパイプライン（ギャラリー走査、特徴量マージ、類似度、Excel）
  core/eval/               精度の検証（insightface 形式 .bin の読込、10 分割交差検証、公表値）
tests/                     pytest（GPU・重み不要）
```

## 実装メモ

- 日本語を含むフォルダ名・ファイル名に対応しています（画像の保存・読込は `core/common/imageio.py` でバイト列経由にして、Windows の cv2 が非 ASCII パスを扱えない問題を回避）。
- 設定は ワークスペースのルートフォルダの `settings.json` に保存されます（Git 管理外）。
- 各モデルは公式の推論手順・重みに合わせて実装し、配布重みの全キー一致の確認と、CFP-FP での公表値との突き合わせ（「[精度の検証（eval）](#精度の検証eval)」）を行っています。LFW / CPLFW 等の `.bin` を `dataset/` に追加すれば同じコマンドで測れます。
