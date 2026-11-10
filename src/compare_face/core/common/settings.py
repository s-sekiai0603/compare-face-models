"""前回指定したフォルダ・抽出頻度・レポート条件の保存/読込（ワークスペース直下の settings.json）。"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_INTERVAL_SEC = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # ワークスペースのルートフォルダ
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

DEFAULTS = {
    "video_root": "",
    "output_root": "",
    "interval_sec": DEFAULT_INTERVAL_SEC,
    "estimate_foot": False,          # 抽出: 足首が取れない場合は足元を動きから推測する
    "fisheye_correct": False,        # 抽出: 映像のフレームを魚眼補正してから処理する
    "extract_camera_dir": "",        # 抽出: カメラ設定フォルダ（<カメラID>.json。魚眼補正に使う）
    "report_mode": "顔画像指定",
    "report_query_path": "",
    "report_track_dir": "",
    "report_gallery_dir": "",
    "report_output_dir": "",
    "report_show_body": True,
    "report_include_old": False,
    "report_min_face_conf": 0.0,
    "report_min_norm": 0.0,
    "report_min_face_width": 0,
    # 顔の採用基準の既定 = 抽出時と同じ値（extract/face_crop.py の MIN_EYE_RATIO / MIN_CROP_SIDE。common から extract を
    # import しないよう値を直接書く。食い違いは tests/test_report.py で検出する）
    "report_face_gate": True,                 # 骨格ルール（鼻+片目 / 下向きでない）を適用
    "report_min_eye_ratio": 0.05,             # 両目間隔比のしきい値
    "report_min_crop_side": 60,               # 切り出し短辺(px)のしきい値
    "report_max_shoulder_reverse": 0.0,       # 肩の左右逆転量のしきい値（骨格ルール有効時のみ。0 で判定しない）
    "report_model": "AdaFace",
    "report_landmark": "なし",
    "report_pose": "なし",
    "report_limit": 500,
    "report_targets": [],
    "report_merge_sec": 0.0,
    "query_set_name": "",            # 検索元セット検索: 前回選んだセット名（顔画像出力先\検索元\<セット名>）
    "query_set_gallery_dir": "",
    "query_set_output_dir": "",
    "query_set_persons": "",         # 本人の追跡IDの追加指定（1 行 1 件「カメラID|映像ファイル名|追跡ID」）
}


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    data = dict(DEFAULTS)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update({k: loaded[k] for k in data if k in loaded})
    except (OSError, ValueError):
        pass
    return data


def save_settings(settings: dict, path: Path = SETTINGS_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---- 顔サイズ別精度検証画面の設定（画面専用のファイル。入力内容をすべて保存して次回起動時に復元する） ----
VERIFY_FACE_SIZE_SETTINGS_PATH = PROJECT_ROOT / "verify_face_size_screen_setting.json"
VERIFY_FACE_SIZE_TRACK_COUNT = 10  # 検証対象人物の追跡IDフォルダの数

VERIFY_FACE_SIZE_DEFAULTS = {
    "output_mode": "similarity",      # similarity（同一人物の類似度）/ misidentify（誤識別の枚数）/ rank（類似度順位）/ both（すべて）
    "track_dirs": [""] * VERIFY_FACE_SIZE_TRACK_COUNT,
    "gallery_dir": "",                # 比較対象フォルダ（誤識別となる画像の枚数レポートで使う）
    "output_dir": "",
    "camera_image_dir": "",           # カメラ画像フォルダ（<カメラID>.png。距離の算出に使う画像サイズをここの画像から取る）
    "camera_dir": "",                 # カメラ設定フォルダ（射影変換設定抽出画面の出力先。<カメラID>.json）。レポートの「距離」の算出に使う
    "model": "AdaFace",
    "landmark": "なし",
    "face_gate": True,
    "min_eye_ratio": 0.05,
    "min_crop_side": 60,
    "max_shoulder_reverse": 0.0,
    "grayscale": False,               # 特徴量抽出前に顔画像をグレースケール化する
    "align168": False,                # 168×168 でアラインしてから 112×112 に縮小して特徴量化する
}


def load_verify_face_size_settings(path: Path = VERIFY_FACE_SIZE_SETTINGS_PATH) -> dict:
    data = {**VERIFY_FACE_SIZE_DEFAULTS, "track_dirs": list(VERIFY_FACE_SIZE_DEFAULTS["track_dirs"])}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update({k: loaded[k] for k in data if k in loaded})
    except (OSError, ValueError):
        pass
    dirs = [str(d) for d in data["track_dirs"]] if isinstance(data["track_dirs"], list) else []
    data["track_dirs"] = (dirs + [""] * VERIFY_FACE_SIZE_TRACK_COUNT)[:VERIFY_FACE_SIZE_TRACK_COUNT]
    return data


def save_verify_face_size_settings(settings: dict, path: Path = VERIFY_FACE_SIZE_SETTINGS_PATH) -> None:
    save_settings(settings, path)


# ---- 指定顔サイズマッピング画面の設定（画面専用のファイル。入力内容を保存して次回起動時に復元する） ----
FACE_SIZE_MAP_SETTINGS_PATH = PROJECT_ROOT / "face_size_map_screen_setting.json"

FACE_SIZE_MAP_DEFAULTS = {
    "face_height": 80,                                               # 顔 bbox の高さ(px)。この値以上の顔の足元をプロットする
    "gallery_dir": "",                                               # 比較対象フォルダ
    "camera_dir": str(PROJECT_ROOT / "debug" / "cam_pics" / "org"),  # カメラ画像フォルダ（ファイル名 = カメラID）
    "output_dir": str(PROJECT_ROOT / "debug" / "cam_pics" / "out"),  # 結果出力先フォルダ
}


def load_face_size_map_settings(path: Path = FACE_SIZE_MAP_SETTINGS_PATH) -> dict:
    data = dict(FACE_SIZE_MAP_DEFAULTS)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update({k: loaded[k] for k in data if k in loaded})
    except (OSError, ValueError):
        pass
    return data


def save_face_size_map_settings(settings: dict, path: Path = FACE_SIZE_MAP_SETTINGS_PATH) -> None:
    save_settings(settings, path)


# ---- 距離・向き・精度分析画面の設定（画面専用のファイル。最後に選んだ Excel を保存して次回起動時に復元する） ----
DISTANCE_POSE_CHART_SETTINGS_PATH = PROJECT_ROOT / "distance_pose_chart_screen_setting.json"

DISTANCE_POSE_CHART_DEFAULTS = {
    "excel_path": "",    # チャートを追加する Excel（顔サイズ別精度検証のレポート）
    "split_lr": False,   # 顔の向きをチャートの左右に振り分ける（横軸を中心 0 m の左右にとる）
    "unit_m": 0.0,       # 距離の集計単位(m)。0 = 集計しない（画像 1 枚ずつプロット）
    "all_cameras": True,     # 出力対象: すべてのカメラを 1 チャートにプロット
    "per_camera": False,     # 出力対象: カメラIDごとに 1 チャートずつプロット
}


def load_distance_pose_chart_settings(path: Path = DISTANCE_POSE_CHART_SETTINGS_PATH) -> dict:
    data = dict(DISTANCE_POSE_CHART_DEFAULTS)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update({k: loaded[k] for k in data if k in loaded})
    except (OSError, ValueError):
        pass
    return data


def save_distance_pose_chart_settings(settings: dict, path: Path = DISTANCE_POSE_CHART_SETTINGS_PATH) -> None:
    save_settings(settings, path)


# ---- 射影変換設定抽出画面の設定（画面専用のファイル。最後に選んだ画像・出力先を保存して次回起動時に復元する） ----
CAMERA_PROJECTION_SETTINGS_PATH = PROJECT_ROOT / "camera_projection_screen_setting.json"

CAMERA_PROJECTION_DEFAULTS = {
    "image_path": "",                                                       # カメラ画像ファイル（ファイル名 = カメラID）
    "output_dir": str(PROJECT_ROOT / "debug" / "cam_pics" / "settings"),    # 設定ファイル出力先フォルダ
    "show_guides": False,                                                   # 消失点・地平線を表示
    "antialias": False,                                                     # アンチエイリアス
    "show_points": True,                                                    # 長方形 4 点の表示
}
CAMERA_PROJECTION_IMAGE_DIR = PROJECT_ROOT / "debug" / "cam_pics" / "org"   # 画像選択時の既定フォルダ


def load_camera_projection_settings(path: Path = CAMERA_PROJECTION_SETTINGS_PATH) -> dict:
    data = dict(CAMERA_PROJECTION_DEFAULTS)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update({k: loaded[k] for k in data if k in loaded})
    except (OSError, ValueError):
        pass
    for key in ("show_guides", "antialias", "show_points"):
        data[key] = data[key] is True
    return data


def save_camera_projection_settings(settings: dict, path: Path = CAMERA_PROJECTION_SETTINGS_PATH) -> None:
    save_settings(settings, path)
