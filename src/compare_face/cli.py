"""コマンドライン実行（GUI を使わない顔画像抽出・レポート出力・モデル一覧）。

  python -m compare_face                    GUI を起動
  python -m compare_face list-models        登録済みモデルと重みの状態を表示
  python -m compare_face extract ...        映像から顔画像を抽出
  python -m compare_face report ...         顔特徴量比較レポートを出力
  python -m compare_face query-set ...      検索元セット（コピーした顔画像の集まり）で総当たり・全体検索
  python -m compare_face eval ...           検証セット（lfw.bin 等）で 1:1 照合精度を測る
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_gate_args(parser: argparse.ArgumentParser) -> None:
    """顔の採用基準（抽出時の顔ゲート・品質ガードのレポート時の再判定）。report と query-set で共通。"""
    from .core.extract.face_crop import MIN_CROP_SIDE, MIN_EYE_RATIO
    parser.add_argument("--no-face-gate", action="store_true",
                        help="顔の採用基準: 骨格ルール（鼻+片目が見える / 下向きでない）を適用しない（既定は抽出時と同じく適用）")
    parser.add_argument("--min-eye-ratio", type=float, default=MIN_EYE_RATIO,
                        help=f"顔の採用基準: 両目間隔÷顔幅がこの値未満の画像（横顔）を使わない（既定 {MIN_EYE_RATIO:g} = 抽出時と同じ。0 で判定しない）")
    parser.add_argument("--min-crop-side", type=float, default=MIN_CROP_SIDE,
                        help=f"顔の採用基準: 切り出し短辺(px)がこの値未満の画像を使わない（既定 {MIN_CROP_SIDE:g} = 抽出時と同じ。0 で判定しない）")
    parser.add_argument("--accept-all", action="store_true",
                        help="顔の採用基準を適用せず、抽出した顔を全て使う（--no-face-gate --min-eye-ratio 0 --min-crop-side 0 と同じ）")


def _gate_filter(args):
    from .core.report.gate import GateFilter
    return (GateFilter.accept_all() if args.accept_all
            else GateFilter(not args.no_face_gate, args.min_eye_ratio, args.min_crop_side))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compare_face", description="顔識別モデル比較ツール（引数なしで GUI 起動）")
    sub = p.add_subparsers(dest="command")

    e = sub.add_parser("extract", help="映像から顔画像を抽出する")
    e.add_argument("--video-root", required=True, help=r"映像ルートフォルダ（ルート\ロケーション\カメラID\*.mp4）")
    e.add_argument("--out", required=True, help="顔画像出力先ルートフォルダ")
    e.add_argument("--interval", type=float, default=0.5, help="抽出頻度(秒)。追跡IDごとにこの間隔で出力（既定 0.5）")
    e.add_argument("--video", action="append", default=[],
                   help="処理する映像ファイル名（複数指定可）。省略時はルート配下の全映像")

    r = sub.add_parser("report", help="顔特徴量比較レポートを出力する")
    r.add_argument("--query", required=True, help="クエリ顔画像(_face.jpg) または追跡IDフォルダ")
    r.add_argument("--gallery", required=True, help="比較対象フォルダ（顔画像出力先と同じ構成）")
    r.add_argument("--out", required=True, help="レポート出力先フォルダ")
    r.add_argument("--model", default="AdaFace", help="顔特徴量抽出モデル名（list-models で確認）")
    r.add_argument("--landmark", default="なし", help="顔ランドマークモデル名（list-models で確認）")
    r.add_argument("--limit", type=int, default=500, help="各シートに出力する上位件数")
    r.add_argument("--merge-sec", type=float, default=0.0, help="追跡ID指定時の特徴量マージ単位(秒)。0 でマージなし")
    r.add_argument("--no-body", action="store_true", help="レポートに全身画像を載せない")
    r.add_argument("--pose", default="なし", help="頭部姿勢モデル名（list-models で確認）。指定すると情報欄に yaw/pitch/roll を表示")
    r.add_argument("--target", action="append", default=[],
                   help="探したい人物「カメラID|映像ファイル名|追跡ID」（複数指定可。空欄の項目は不問）")
    r.add_argument("--no-cache", action="store_true",
                   help="比較対象の特徴量キャッシュ（比較対象フォルダ内 .embedding_cache/）を使わない")
    r.add_argument("--include-old", action="store_true",
                   help="追跡IDフォルダ内の old フォルダに移した画像も、クエリ・比較対象に元の追跡IDの画像として含める（選別なしで比較）")
    r.add_argument("--min-face-conf", type=float, default=0.0,
                   help="クエリ自動選別: 顔検出の信頼度がこの値未満の画像を使わない（0 で無効。目安 0.65）")
    r.add_argument("--min-norm", type=float, default=0.0,
                   help="クエリ自動選別: 品質ノルム（AdaFace の特徴量ノルム）がこの値未満の画像を使わない（0 で無効。目安 17）")
    r.add_argument("--min-face-width", type=float, default=0.0,
                   help="クエリ自動選別: 顔枠の幅(px)がこの値未満の画像を使わない（0 で無効。目安 30）")
    _add_gate_args(r)
    r.add_argument("--exclude-dir", action="append", default=[],
                   help="クエリ自身に加えて比較対象から外す追跡IDフォルダ（複数指定可。クエリ画像を別フォルダに集めたときの元の追跡IDなど）")
    r.add_argument("--clahe", action="store_true",
                   help="アライン後の顔画像に CLAHE（明るさ補正）を掛けてから特徴量化する（premises-security の AdaFace 前処理と同じ。"
                        "比較対象の特徴量キャッシュは別ファイルになる）")

    qs = sub.add_parser("query-set", help="検索元セット（顔画像出力先\\検索元\\<セット名> にコピーした顔画像）の各画像で総当たり・全体検索をして Excel に出力する")
    qs.add_argument("--set", required=True, help="検索元セットのフォルダ（比較対象フォルダの _face.jpg をコピーしたフォルダ。サブフォルダ可）")
    qs.add_argument("--gallery", required=True, help="比較対象フォルダ（顔画像出力先と同じ構成）")
    qs.add_argument("--out", required=True, help="Excel 出力先フォルダ")
    qs.add_argument("--model", default="AdaFace", help="顔特徴量抽出モデル名（list-models で確認）")
    qs.add_argument("--landmark", default="なし", help="顔ランドマークモデル名（list-models で確認）")
    qs.add_argument("--pose", default="なし", help="頭部姿勢モデル名（list-models で確認）。指定すると検索元の yaw/pitch/roll を表示")
    qs.add_argument("--person", action="append", default=[],
                    help="本人の追跡ID「カメラID|映像ファイル名|追跡ID」を追加（複数指定可。空欄の項目は不問。検索元の元画像の追跡IDは自動で本人）")
    qs.add_argument("--no-cache", action="store_true", help="比較対象の特徴量キャッシュを使わない")
    qs.add_argument("--include-old", action="store_true", help="追跡IDフォルダ内の old フォルダに移した画像も比較対象に含める")
    _add_gate_args(qs)

    v = sub.add_parser("eval", help="検証セット（insightface 形式の .bin）で顔特徴量モデルの 1:1 照合精度を測る")
    v.add_argument("--bin", action="append", default=None,
                   help="検証セットの .bin（複数指定可）。省略時は dataset/ 配下の全 .bin")
    v.add_argument("--model", action="append", default=None,
                   help="顔特徴量抽出モデル名（複数指定可）。省略時は画面に出る全モデル")
    v.add_argument("--landmark", action="append", default=None,
                   help="顔ランドマークモデル名（複数指定可）。省略時は「なし」（.bin はアライン済みのため）")
    v.add_argument("--out", help="結果 CSV の出力先。省略時は dataset/eval_<日時>.csv")
    v.add_argument("--limit", type=int, default=0, help="等間隔に選んだこのペア数だけで評価する（動作確認用。0 で全ペア）")

    sub.add_parser("list-models", help="登録済みのモデルと重みの状態を表示する")

    rv = sub.add_parser("revalidate", help="抽出済みの顔画像に無効判定（顔枠に本人の鼻点が無い = 別人の顔）を後から適用する")
    rv.add_argument("--faces", required=True, help="顔画像出力先ルート（配下の全追跡ID JSON を更新。既に無効の画像は触らない）")

    d = sub.add_parser("download", help="モデルの重みを取得する（自動取得に対応したモデルのみ）")
    d.add_argument("--model", action="append", default=[], help="顔特徴量抽出モデル名（複数指定可。省略時は全モデル）")
    d.add_argument("--landmark", action="append", default=[], help="顔ランドマークモデル名（複数指定可。省略時は全モデル）")
    d.add_argument("--pose", action="append", default=[], help="頭部姿勢モデル名（複数指定可。省略時は全モデル）")
    return p


def _download(args) -> int:
    from .core.features import EXTRACTORS, WeightsUnavailable
    from .core.landmarks import ALIGNERS
    from .core.pose import POSE_OPTIONS, get_pose_estimator
    names_e = args.model or list(EXTRACTORS)
    names_l = args.landmark or list(ALIGNERS)
    names_p = args.pose or [n for n, v in POSE_OPTIONS.items() if v]
    pose_registry = {n: (lambda n=n: get_pose_estimator(n)) for n in POSE_OPTIONS if POSE_OPTIONS[n]}
    failed = 0
    for registry, names in ((EXTRACTORS, names_e), (ALIGNERS, names_l), (pose_registry, names_p)):
        for name in names:
            if name not in registry:
                print(f"未登録のモデル名: {name}", file=sys.stderr)
                failed += 1
                continue
            obj = registry[name]()
            if obj.weights_ready():
                print(f"{name}: 取得済み")
                continue
            try:
                obj.download_weights(log=print)
                print(f"{name}: {'取得完了' if obj.weights_ready() else '取得できませんでした'}")
                failed += 0 if obj.weights_ready() else 1
            except WeightsUnavailable as e:
                print(f"{name}: 自動取得できません\n{e}")
                failed += 1
    return 1 if failed else 0


def _list_models() -> int:
    from .core.features import EXTRACTORS
    from .core.landmarks import ALIGNERS
    from .core.common.weights import DETECTION_DIR
    from .core.extract.face_detector import FACE_MODEL_CANDIDATES

    def status(obj) -> str:
        return "重みあり" if obj.weights_ready() else "重みなし（要配置）"

    print("[顔特徴量抽出モデル]（「非表示」は画面に出さないが --model では使える）")
    for name, cls in EXTRACTORS.items():
        obj = cls()
        print(f"  {name}: {obj.description} ... {status(obj)}{'（非表示）' if cls.hidden else ''}")
    print("[顔ランドマークモデル]")
    for name, cls in ALIGNERS.items():
        obj = cls()
        print(f"  {name}: {obj.description} ... {status(obj)}")
    from .core.pose import POSE_OPTIONS, get_pose_estimator
    print("[頭部姿勢モデル]（レポートの情報欄に yaw/pitch/roll を表示）")
    for name in POSE_OPTIONS:
        est = get_pose_estimator(name)
        print(f"  {name}: {'表示しない' if est is None else est.weights_path().name + ' ... ' + status(est)}")
    print("[顔検出モデル]（上から順に試し、最初に読み込めたものを使う。無ければ .pt を自動取得）")
    for name in FACE_MODEL_CANDIDATES:
        print(f"  {name}: {'あり' if (DETECTION_DIR / name).is_file() else 'なし'}")
    return 0


def _extract(args) -> int:
    from .core.extract.pipeline import run_extraction
    from .core.extract.video_scanner import scan_videos
    videos = scan_videos(Path(args.video_root))
    if args.video:
        wanted = set(args.video)
        videos = [v for v in videos if v.file_name in wanted]
    if not videos:
        print("処理対象の映像がありません", file=sys.stderr)
        return 1
    results, run_dir = run_extraction(videos, Path(args.out), args.interval, status=print,
                                      progress=lambda i, n, f, t: None)
    faces = sum(r.image_count for r in results)
    tracks = sum(r.track_count for r in results)
    print(f"完了: 映像 {len(results)} 本 / 追跡ID {tracks} / 顔画像 {faces} 枚"
          f"（うち採用基準外 {sum(r.rejected_count for r in results)} 枚、無効 {sum(r.invalid_count for r in results)} 枚）"
          f"\n出力先: {run_dir}")
    return 0


def parse_target(text: str) -> tuple[str, str, str]:
    parts = [s.strip() for s in text.split("|")] + ["", "", ""]
    return parts[0], parts[1], parts[2]


def _report(args) -> int:
    from .core.report.pipeline import run_report
    from .core.report.excel import MODE_IMAGE, MODE_TRACK
    from .core.report.quality import QueryFilter
    query = Path(args.query)
    mode = MODE_TRACK if query.is_dir() else MODE_IMAGE
    out, _ = run_report(model_name=args.model, landmark_name=args.landmark, mode=mode, query_path=query,
                        gallery_dir=Path(args.gallery), output_dir=Path(args.out), show_body=not args.no_body,
                        limit=args.limit, targets=[parse_target(t) for t in args.target],
                        merge_sec=0.0 if mode == MODE_IMAGE else args.merge_sec, pose_name=args.pose,
                        use_cache=not args.no_cache, include_excluded=args.include_old,
                        query_filter=QueryFilter(args.min_face_conf, args.min_norm, args.min_face_width),
                        exclude_dirs=[Path(p) for p in args.exclude_dir], clahe=args.clahe, gate_filter=_gate_filter(args))
    print(f"レポートを出力しました: {out}")
    return 0


def _query_set(args) -> int:
    from .core.report.query_set import run_query_set_search
    out, _ = run_query_set_search(set_dir=Path(args.set), gallery_dir=Path(args.gallery), output_dir=Path(args.out),
                                  model_name=args.model, landmark_name=args.landmark, pose_name=args.pose,
                                  include_excluded=args.include_old, gate_filter=_gate_filter(args),
                                  own_specs=[parse_target(t) for t in args.person], use_cache=not args.no_cache)
    print(f"Excel を出力しました: {out}")
    return 0


def _eval(args) -> int:
    from .core.eval.pipeline import default_bins, default_csv_path, run_eval
    from .core.features import visible_extractors
    bins = [Path(b) for b in args.bin] if args.bin else default_bins()
    out = Path(args.out) if args.out else default_csv_path()
    rows = run_eval(bins, args.model or list(visible_extractors()), args.landmark or ["なし"], out, args.limit)
    print(f"完了: {len(rows)} 件 / 結果 CSV: {out}")
    return 0


def _revalidate(args) -> int:
    from .core.extract.validity import revalidate_tree
    n_json, checked, newly = revalidate_tree(Path(args.faces))
    print(f"追跡ID JSON {n_json} 件 / 判定した画像 {checked} 枚 / 無効の増減 {newly:+d} 枚（顔枠に本人の鼻点が無い。"
          "以前の同ルールによる無効は判定し直し）")
    return 0


def _configure_console() -> None:
    """パイプ/リダイレクト先（自動実行のログ等）へは UTF-8 で書き、端末ではその端末の文字コードを使う。
    いずれも変換できない文字で落ちないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        from .app import main as gui_main
        return gui_main()
    _configure_console()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-models":
            return _list_models()
        if args.command == "extract":
            return _extract(args)
        if args.command == "report":
            return _report(args)
        if args.command == "query-set":
            return _query_set(args)
        if args.command == "download":
            return _download(args)
        if args.command == "eval":
            return _eval(args)
        if args.command == "revalidate":
            return _revalidate(args)
    except Exception as e:  # noqa: BLE001  コマンド実行は要因を表示して終了コードで知らせる
        print(f"エラー: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    build_parser().print_help()
    return 2
