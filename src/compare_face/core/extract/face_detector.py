"""YOLOv8-Face による顔検出（premises-security core/detection/yolo_face.py 相当）。

検出モデルは models/ に置かれたものを使う。候補（FACE_ENGINE_CANDIDATES → .pt）を順に試し、
最初に読み込めたものを使う。TensorRT エンジンは GPU・TensorRT バージョン依存で読み込めないことが
あるため、その場合は .pt（無ければ公開リポジトリから自動取得）へフォールバックし、TensorRT が
使える環境では本ツール用のエンジン（FACE_ENGINE_AUTO）に変換して使う。既存のエンジンは上書きしない。
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..common.weights import DETECTION_DIR as MODELS_DIR  # 顔検出モデルは models/detection/ に置く
CONF_TH = 0.30   # premises-security YOLO_FACE_CONF
NMS_IOU = 0.60   # premises-security YOLO_FACE_IOU
IMGSZ = 640
MAX_DETS = 200
MAX_BATCH = 128  # b128 エンジンの最大バッチ

FACE_PT_NAME = "yolov8s-face-lindevs.pt"
FACE_PT_URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt"
FACE_ENGINE_AUTO = f"yolov8s-face-lindevs-b{MAX_BATCH}-auto.engine"  # 本ツールが .pt から変換したエンジン
FACE_ENGINE_CANDIDATES = (
    "yolov8s-face-lindevs-b128.engine",  # premises-security 由来の TensorRT エンジン（最大バッチ128）
    "yolov8s-face-lindevs.engine",
    FACE_ENGINE_AUTO,
)
FACE_MODEL_CANDIDATES = FACE_ENGINE_CANDIDATES + (FACE_PT_NAME,)


@dataclass
class FaceDetections:
    boxes: np.ndarray   # (N,4) xyxy（入力画像の座標）
    scores: np.ndarray  # (N,)

    @staticmethod
    def empty() -> "FaceDetections":
        return FaceDetections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


def find_face_model(models_dir: Path = MODELS_DIR) -> Path | None:
    """配置済みの候補のうち優先順位が最も高いファイル（読み込めるかは確認しない）。"""
    for name in FACE_MODEL_CANDIDATES:
        p = models_dir / name
        if p.is_file():
            return p
    return None


def face_model_missing_message(models_dir: Path = MODELS_DIR) -> str:
    return ("顔検出モデルが見つかりません。次のいずれかを配置してください:\n"
            + "\n".join(f"  {models_dir / n}" for n in FACE_MODEL_CANDIDATES)
            + f"\n（.pt は {FACE_PT_URL} から自動取得を試みます）")


class FaceDetector:
    def __init__(self, model_path: str):
        from ultralytics import YOLO
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path), task="detect")
        self.model_name = Path(str(model_path)).name
        self.backend = "TensorRT" if str(model_path).lower().endswith(".engine") else "PyTorch"
        self._warm_up()  # エンジンのバージョン不整合などは初回推論まで分からないため、ここで確かめる
        self.describe()

    def _warm_up(self) -> None:
        self.model.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), conf=CONF_TH, iou=NMS_IOU, imgsz=IMGSZ,
                           max_det=MAX_DETS, device=self.device, verbose=False)

    def describe(self) -> str:
        import torch
        dev = "CPU" if self.device == "cpu" else f"GPU (cuda:{self.device}, {torch.cuda.get_device_name(self.device)})"
        msg = (f"[推論環境] 顔検出: デバイス: {dev} / フレームワーク: {self.backend}"
               f" / conf={CONF_TH} iou={NMS_IOU} imgsz={IMGSZ} / モデル: {self.model_name}")
        print(msg, flush=True)
        return msg

    def detect(self, image_bgr: np.ndarray) -> FaceDetections:
        return self.detect_batch([image_bgr])[0]

    def detect_batch(self, images: list[np.ndarray]) -> list[FaceDetections]:
        """複数画像（サイズ不揃い可）を MAX_BATCH 単位でまとめて推論する。結果は入力と同じ順序。"""
        out: list[FaceDetections] = []
        for i in range(0, len(images), MAX_BATCH):
            results = self.model.predict(
                images[i:i + MAX_BATCH], conf=CONF_TH, iou=NMS_IOU, imgsz=IMGSZ,
                max_det=MAX_DETS, device=self.device, verbose=False)
            out.extend(self._to_detections(r) for r in results)
        return out

    @staticmethod
    def _to_detections(res) -> FaceDetections:
        if res.boxes is None or len(res.boxes) == 0:
            return FaceDetections.empty()
        return FaceDetections(res.boxes.xyxy.cpu().numpy().astype(np.float32),
                              res.boxes.conf.cpu().numpy().astype(np.float32))


def download_face_pt(dst: Path, log: Callable[[str], None] = print, url: str = FACE_PT_URL) -> Path:
    """公開リポジトリから .pt を取得する（.part に書いてから改名）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"{dst.name} をダウンロードしています: {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, str(tmp))
    if not tmp.is_file() or tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ダウンロード結果が不正です: {url}")
    tmp.replace(dst)
    return dst


def export_face_engine(pt_path: Path, engine_path: Path, log: Callable[[str], None] = print) -> Path:
    """.pt を最大バッチ MAX_BATCH の動的バッチ TensorRT エンジンへ変換する（初回のみ数分）。

    ultralytics は .pt と同じ場所に .onnx/.engine を書くため、既存のエンジンを上書きしないよう
    一時フォルダにコピーした .pt から変換し、出来上がったエンジンだけを engine_path へ移す。"""
    import shutil
    import tempfile
    from ultralytics import YOLO
    log(f"{pt_path.name} を TensorRT エンジンへ変換しています（最大バッチ {MAX_BATCH}。初回のみ時間がかかります）...")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="face_export_", dir=str(engine_path.parent)) as tmp:
        tmp_pt = Path(tmp) / pt_path.name
        shutil.copy2(pt_path, tmp_pt)
        exported = Path(YOLO(str(tmp_pt)).export(format="engine", half=True, device=0, batch=MAX_BATCH,
                                                 dynamic=True, imgsz=IMGSZ))
        if not exported.is_file():
            raise RuntimeError(f"変換結果が見つかりません: {exported}")
        shutil.move(str(exported), str(engine_path))
    return engine_path


def _short_error(e: Exception) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0][:200] if text else ''}"


def prepare_face_detector(log: Callable[[str], None] = print, models_dir: Path = MODELS_DIR) -> FaceDetector:
    """読み込める顔検出モデルを用意して FaceDetector を返す。

    1. models/ のエンジン候補を順に試す（読み込めないものはログに出して次へ）
    2. どれも使えなければ .pt を使う（無ければ自動取得）。TensorRT が使える環境では
       本ツール用エンジン（FACE_ENGINE_AUTO）へ変換して使い、変換に失敗したら .pt のまま使う
    """
    from .pose_model import tensorrt_available

    for name in FACE_ENGINE_CANDIDATES:
        p = models_dir / name
        if not p.is_file():
            continue
        try:
            return FaceDetector(str(p))
        except Exception as e:  # noqa: BLE001  バージョン不整合など。次の候補へ
            log(f"{name} を読み込めないため次の候補を試します（{_short_error(e)}）")
    pt = models_dir / FACE_PT_NAME
    if not pt.is_file():
        download_face_pt(pt, log)
    if tensorrt_available():
        try:
            return FaceDetector(str(export_face_engine(pt, models_dir / FACE_ENGINE_AUTO, log)))
        except Exception as e:  # noqa: BLE001  変換失敗時は .pt にフォールバック
            log(f"TensorRT 変換に失敗したため .pt を使用します（{_short_error(e)}）")
    return FaceDetector(str(pt))
