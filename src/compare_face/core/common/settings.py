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
    "output_mode": "similarity",      # similarity（同一人物の類似度）/ misidentify（誤識別の枚数）/ both
    "track_dirs": [""] * VERIFY_FACE_SIZE_TRACK_COUNT,
    "gallery_dir": "",                # 比較対象フォルダ（誤識別となる画像の枚数レポートで使う）
    "output_dir": "",
    "model": "AdaFace",
    "landmark": "なし",
    "face_gate": True,
    "min_eye_ratio": 0.05,
    "min_crop_side": 60,
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
