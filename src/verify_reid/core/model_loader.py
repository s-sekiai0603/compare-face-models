"""YOLOv8-pose の重み準備。

TensorRT/CUDA が使えれば .pt を .engine に変換して使用し、
使えなければ ultralytics 標準 .pt を自動DL/配置して使用する。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .pose_detector import BATCH_SIZE

MODEL_NAME = "yolov8l-pose"
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def tensorrt_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    return importlib.util.find_spec("tensorrt") is not None


def prepare_model_path(log=print) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = MODELS_DIR / f"{MODEL_NAME}.pt"
    engine_path = MODELS_DIR / f"{MODEL_NAME}-b{BATCH_SIZE}.engine"

    if tensorrt_available():
        if engine_path.exists():
            return engine_path
        try:
            if not pt_path.exists():
                log(f"{pt_path.name} をダウンロードしています...")
                attempt_download_asset(str(pt_path))
            from ultralytics import YOLO
            log("TensorRT エンジンへ変換しています（初回のみ時間がかかります）...")
            exported = YOLO(str(pt_path)).export(format="engine", half=True, device=0,
                                                batch=BATCH_SIZE, dynamic=True)
            exported = Path(exported)
            if exported != engine_path and exported.exists():
                exported.replace(engine_path)
            return engine_path
        except Exception as e:  # 変換失敗時は .pt にフォールバック
            log(f"TensorRT変換に失敗したため .pt を使用します: {e}")

    if not pt_path.exists():
        log(f"{pt_path.name} をダウンロードしています...")
        attempt_download_asset(str(pt_path))
    return pt_path
