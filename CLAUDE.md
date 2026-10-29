# CLAUDE.md

このリポジトリで Claude Code が作業するときのルール。

## Git 操作のルール（最重要）

- **絶対に自動で `git push` しないこと。** push はユーザーが明示的に「push して」と指示したときだけ行う。
  コミットを作った後も push はしない。「コミットして」は「push して」を含まない。
- push 先は `origin` (`https://github.com/s-sekiai0603/compare-face-models.git`) のみ。
  `main` ブランチの追跡先と `remote.pushDefault` は `origin` に設定済み。
- **移行前の旧リポジトリには絶対に push しないこと。** 旧リポジトリとの接続は意図的に切ってある。
  リモートとして再追加しない（`origin` 以外のリモートを足さない）。
- `origin` のリモート設定（URL、名前）を変更・削除しないこと。
- `git push --force` / `--force-with-lease` は、ユーザーが明示的に指示した場合を除き禁止。
- `git commit` もユーザーの指示があったときだけ行う。勝手にコミットしない。
- `models/` 配下（重み・エンジン。合計数 GB）は `.gitignore` で除外している。解除しない。

## プロジェクト概要（compare-face-models）

顔識別モデル比較ツール。映像から顔画像を抽出し、顔特徴量モデル／顔ランドマークモデル／頭部姿勢モデルを
切り替えて類似度比較レポート（Excel）を出力する。抽出と顔ゲートは隣のリポジトリ `../premises-security` の
顔識別（YOLOv8-Face → 骨格による顔ゲート → 適応マージン）と同じにしてある。

- Python 3.10〜3.12。依存管理は `uv`（`pyproject.toml` / `uv.lock`）。`uv sync` で `.venv` を作る。
- 起動: `start.bat` または `uv run python -m compare_face`。CLI: `extract` / `report` / `query-set` / `eval` / `revalidate` / `download` / `list-models`。
- テスト: `uv run pytest`（GPU・モデル重みなしで全件通ること。重みがあれば strict 読込の確認も走る）。
- `reid/`（改修前の外見識別(reid)版のソース）は手元に参照用として残しているだけで Git 管理外（`.gitignore`）。編集しない。
- 設定は `settings.json`（Git 管理外）。
- `dataset/` は精度検証用の insightface 形式 `.bin`（cfp_fp.bin 等）の置き場所（Git 管理外。`.gitkeep` のみ管理）。
  `eval` コマンドの結果 CSV もここに出る。README の「精度の検証（eval）」の結果表は `eval` の実測値で更新する。

### ソース構成（`src/compare_face/`）

| 場所 | 役割 |
|---|---|
| `cli.py` / `app.py` / `ui/` | コマンドライン、GUI（PySide6）。処理本体は持たない。`ui/widgets.py` は画面共通の部品（顔の採用基準の行、モデル選択、重みの確認）、`ui/query_set_screen.py` は検索元セット検索画面 |
| `core/common/` | 共通部品: `weights.py`（models/ の各フォルダと `WeightsUnavailable`）, `settings.py`, `imageio.py`（日本語パス対応の読み書き）, `phase_log.py`, `face_template.py`（ArcFace 5 点）, `cvlface_repo.py`（Hugging Face リポジトリの直接読込） |
| `core/extract/` | 抽出パイプライン: `pipeline.py`（映像 → 顔画像）, `pose_detector.py` / `pose_model.py`（YOLOv8-pose）, `face_detector.py`（YOLOv8-Face）, `tracker.py`（BoT-SORT）, `face_crop.py`（顔ゲート・適応マージン・小顔 ROI）, `validity.py`, `video_scanner.py` |
| `core/landmarks/` | 顔ランドマークモデルのレジストリ（`base.py` の `FaceAligner` を継承して `@register`）: なし / 2d106det / SCRFD / RetinaFace-R50（`retinaface_r50.py`、Pytorch_Retinaface の移植）/ MTCNN（`mtcnn.py` + `mtcnn_net.py`、AdaFace 公式の移植）/ CVLFace DFA。入力は顔がほぼ全面の切り出しなので、検出器には余白を足してから渡す |
| `core/features/` | 顔特徴量モデルのレジストリ（`base.py` の `FaceFeatureExtractor` を継承して `@register`）: AdaFace / CVLFace / LAFS / FSPFM。`*_net.py` は元コードから移植したネットワーク定義 |
| `core/pose/` | 頭部姿勢のレジストリ（`__init__.py` の `POSE_OPTIONS`）: HopeNet（`hopenet.py`。FSPFM の姿勢入力にも使う）/ 6DRepNet（`sixdrepnet.py`、RepVGG-B1g2 の移植）。レポートの yaw/pitch/roll 表示に使う |
| `core/report/` | レポートパイプライン: `pipeline.py`（アライン → 特徴量 → 類似度 → Excel）, `gallery.py`, `merge.py`, `similarity.py`, `excel.py`, `cache.py`（比較対象の特徴量キャッシュ）, `quality.py`（クエリ自動選別と品質ノルム）, `gate.py`（顔の採用基準 = 抽出時の顔ゲート・品質ガードの再判定）, `query_set.py` / `query_set_excel.py`（検索元セット検索: 顔画像出力先\検索元\<セット名> にコピーした N 枚の総当たりと比較対象全体での順位。元画像はバイト一致で特定） |
| `core/report/verify_face_size.py` / `verify_face_size_excel.py` | 顔サイズ別精度検証（追跡IDフォルダ 1〜10 の顔画像を顔 bbox サイズ順に並べた類似度 Excel。画面は `ui/verify_face_size_screen.py`、入力内容は `verify_face_size_screen_setting.json`（Git 管理外）に保存。誤識別の枚数レポート = 対象者の画像ごとに、指定追跡ID以外（非対象者）のうち列の対象者より似ている画像の数。詳細は同名の `.txt`） |
| `core/eval/` | 精度の検証: `verification.py`（.bin 読込、insightface と同じ 10 分割交差検証）, `pipeline.py`（モデル × ランドマークの一括評価と CSV）, `published.py`（論文・README の公表値と出典） |

### モデル重みの置き場所（`models/`、Git 管理外）

`core/common/weights.py` の `DETECTION_DIR` / `LANDMARK_DIR` / `FEATURE_DIR` / `POSE_DIR` が唯一の定義。
`detection/`（YOLO の重み・エンジン）, `landmark/`（2d106det.onnx, det_10g.onnx, retinaface/, mtcnn/, cvlface_DFA_mobilenet/）,
`feature/`（adaface/, cvlface/, lafs/, fspfm/）, `pose/`（hopenet/, 6drepnet/）。YOLO と AdaFace の重みは premises-security の `models/` と同じファイル。

## 変更時の注意

- モデルを追加するときは README の「モデルの追加方法」の手順に従い、レジストリ登録だけで画面に並ぶようにする。
  画面に出したくないモデルは `hidden = True`（CLI の `--model` では使える）。
- premises-security と同じ判定ロジック（`core/extract/face_crop.py` の定数）を変えるときは README の表も更新する。
- `tensorrt-cu12` は `models/detection/` の `.engine` と同じ 10.16.0.72 に固定している。TensorRT の版を変えると既存エンジンは
  読み込めなくなる（ツールは自動で作り直す/フォールバックするが、ユーザー配置のエンジンは使われなくなる）ので、安易に上げない。
- CVLFace は transformers を使わない（同梱ラッパーがカレントディレクトリ依存で失敗する）。CUDA 拡張 rpe_ops はビルドしない
  （ビルド成功時に sys.exit する実装のため `without_extension_build` で止めている）。同梱の RPE パッケージは import 時に
  `os.chdir()` して戻さないので、`cvlface_repo.import_package` が import 前のカレントディレクトリを復元する。
- LAFS / FSPFM の配布チェックポイントは分類層や optimizer を含む。`load_partfvit_state_dict` / `load_fspfm_state_dicts` で
  推論に必要な部分だけ取り出して strict=True で読む（キー不一致はテストで検出される）。
- `features/`・`landmarks/`・`pose/` は `core/common/weights.py` から import する（互いに import すると循環する）。
- `core/eval/published.py` の公表値は出典（論文の表・公式 README）を確認したうえで入れる。モデルを追加したら対応する行も足す。
- 追跡IDフォルダ内の `old` フォルダ（`gallery.EXCLUDED_DIR_NAMES`）は人が外した画像の置き場。既定ではクエリ・比較対象とも
  除外し、`include_excluded=True`（画面の「old フォルダの画像も含める」/ CLI `--include-old`）で元の追跡IDの画像として戻す。
  old の画像を「追跡ID old」として扱わないこと（`_make_entry` で親フォルダに付け替えている）。
- 比較対象の特徴量は `core/report/cache.py` で比較対象フォルダ内 `.embedding_cache/` にキャッシュされる。前処理・アライン・
  TTA など特徴量の値が変わる変更をしたら `CACHE_SCHEMA` を上げて古いキャッシュを無効化すること。
- 無効判定は `core/extract/validity.py` の 2 ルール（別人の骨格 / 顔枠に本人の鼻点が無い）。クエリ自動選別（顔信頼度・品質ノルム・顔幅）は
  クエリだけに適用し、比較対象には適用しない（人物ごと消えるため）。品質ノルムは識別モデルに関係なく AdaFace で測る。
- 抽出は顔ゲート・品質ガード（採用基準）に外れた顔も保存し、JSON の `gate_ok` / `gate_reason` / `face_visible` / `crop_min_side` に
  判定を記録する。採用される顔の出力タイミングと処理量を変えないため、基準外の顔は採用の間隔内では出さず、採用できる顔が無い間だけ
  基準外の間隔で出す（`pipeline.decide_output`。出力対象の判定 `due` は従来どおり採用の間隔だけで行う）。
  使う・使わないはレポート時の `core/report/gate.py`（`GateFilter`）で決め、クエリ・比較対象の両方に適用する。既定値は抽出時と同じ
  （骨格ルールあり / 両目間隔比 0.05 / 切り出し短辺 60px）なので従来と同じ結果になる。判定値の無い古い抽出結果は通ったとみなす。
  `face_crop.py` の定数を変えるときは `GateFilter` の既定値（同じ定数を参照）と README の表も確認する。
- 検索元セット（`gallery.QUERY_SETS_DIR_NAME` = `検索元`）は顔画像出力先の直下に置くので比較対象フォルダの中に入る。`scan_gallery` は
  このフォルダを常に飛ばす（コピーした画像を比較対象に二重に数えない）。検索元の元画像はバイト一致（サイズ + MD5）で探し、比較対象に
  ある画像は特徴量を再利用する。本人の追跡ID = 元画像の追跡ID + 追加指定で、比較対象に無い追跡IDは順位を出せないので本人に入れない。
- 入力のチャネル順: AdaFace は公式 inference.py どおり BGR、CVLFace / LAFS / FSPFM は RGB。いずれも CFP-FP で両方を
  測って決めた値なので、根拠なく変えない。premises-security の AdaFace 前処理（RGB + CLAHE）を再現したいときは
  `--clahe` を使う（チャネル順だけは BGR のままで、CFP-FP の差は 0.2 ポイント）。
