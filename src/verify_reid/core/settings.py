"""前回指定したフォルダ・抽出頻度の保存/読込。"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_INTERVAL_SEC = 0.5
SETTINGS_PATH = Path.home() / ".verify_appearance_reid" / "settings.json"


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    data = {
        "video_root": "",
        "output_root": "",
        "interval_sec": DEFAULT_INTERVAL_SEC,
        "report_mode": "外見画像指定",
        "report_query_path": "",
        "report_track_dir": "",
        "report_gallery_dir": "",
        "report_output_dir": "",
        "report_skeleton": False,
        "report_model": "OSNet",
    }
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
