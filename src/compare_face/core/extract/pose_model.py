"""YOLOv8-pose の重み準備。

TensorRT/CUDA が使えれば .pt を .engine に変換して使用し、
使えなければ ultralytics 標準 .pt を自動DL/配置して使用する。
.engine は TensorRT のバージョンが変わると読み込めなくなるため、読み込めない場合は作り直す。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

from .pose_detector import BATCH_SIZE, PoseDetector

MODEL_NAME = "yolov8l-pose"
from ..common.weights import DETECTION_DIR as MODELS_DIR  # 人物検出モデルは models/detection/ に置く


def tensorrt_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    return importlib.util.find_spec("tensorrt") is not None


def _ensure_pt(models_dir: Path, log: Callable[[str], None]) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset
    models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = models_dir / f"{MODEL_NAME}.pt"
    if not pt_path.exists():
        log(f"{pt_path.name} をダウンロードしています...")
        attempt_download_asset(str(pt_path))
    return pt_path


def export_pose_engine(pt_path: Path, engine_path: Path, log: Callable[[str], None] = print) -> Path:
    """.pt を最大バッチ BATCH_SIZE の動的バッチ TensorRT エンジンへ変換する（初回のみ数分）。"""
    from ultralytics import YOLO
    log("TensorRT エンジンへ変換しています（初回のみ時間がかかります）...")
    exported = Path(YOLO(str(pt_path)).export(format="engine", half=True, device=0,
                                              batch=BATCH_SIZE, dynamic=True))
    if exported != engine_path and exported.exists():
        exported.replace(engine_path)
    return engine_path


def prepare_model_path(log: Callable[[str], None] = print, models_dir: Path = MODELS_DIR) -> Path:
    """使用する重みのパス（.engine があればそれ、無ければ変換、できなければ .pt）。読み込めるかは確認しない。"""
    pt_path = models_dir / f"{MODEL_NAME}.pt"
    engine_path = models_dir / f"{MODEL_NAME}-b{BATCH_SIZE}.engine"
    if tensorrt_available():
        if engine_path.exists():
            return engine_path
        try:
            return export_pose_engine(_ensure_pt(models_dir, log), engine_path, log)
        except Exception as e:  # 変換失敗時は .pt にフォールバック
            log(f"TensorRT変換に失敗したため .pt を使用します: {e}")
    return _ensure_pt(models_dir, log)


def _short_error(e: Exception) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0][:200] if text else ''}"


def prepare_pose_detector(log: Callable[[str], None] = print, models_dir: Path = MODELS_DIR) -> PoseDetector:
    """読み込める人物検出モデルを用意して PoseDetector を返す。

    既存の .engine が読み込めない（TensorRT のバージョンが変わった等）場合は、本ツールが作った
    エンジンなので削除して作り直す。変換もできなければ .pt を使う。
    """
    path = prepare_model_path(log, models_dir)
    try:
        return PoseDetector(str(path))
    except Exception as e:  # noqa: BLE001
        if path.suffix.lower() != ".engine":
            raise
        log(f"{path.name} を読み込めないため作り直します（{_short_error(e)}）")
        path.unlink(missing_ok=True)
    try:
        return PoseDetector(str(export_pose_engine(_ensure_pt(models_dir, log), path, log)))
    except Exception as e:  # noqa: BLE001
        log(f"TensorRT 変換に失敗したため .pt を使用します（{_short_error(e)}）")
    return PoseDetector(str(_ensure_pt(models_dir, log)))
