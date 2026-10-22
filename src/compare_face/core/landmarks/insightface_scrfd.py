"""InsightFace SCRFD（det_10g.onnx, buffalo_l）による 5 点検出と、insightface の norm_crop と同じアライン。

MS1M-RetinaFace や WebFace4M/12M など、多くの顔認識モデルの学習データは RetinaFace/SCRFD の 5 点
（左目・右目・鼻先・左口角・右口角）を ArcFace テンプレートへ相似変換して 112x112 に整列している。
このアライナはその手順をそのまま再現するため、論文の前処理に最も近い選択肢になる。

元コード: insightface/python-package/insightface/model_zoo/scrfd.py, utils/face_align.py（推論部分のみ移植）。
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..common.weights import LANDMARK_DIR
from .base import AlignResult, FaceAligner, WeightsUnavailable, register, template_5pts
from .insightface_pack import ensure_buffalo_file

MODEL_FILE = "det_10g.onnx"
INPUT_SIZE = (640, 640)   # insightface の既定 det_size
DET_THRESH = 0.5
# 0.5 で見つからないときに下げて再試行する。入力は YOLOv8-Face で顔と確定した切り出しなので誤検出の心配が小さく、
# 監視映像の小さくぼやけた顔（15〜40px を拡大）では 0.05 まで下げて初めて 9 割検出できる
# （実測 51 枚: 0.5 で 17 枚, 0.3 で 27 枚, 0.05 で 46 枚。2d106det との 5 点のずれは 3〜4px で変わらない）。
DET_THRESH_RETRY = 0.05
NMS_THRESH = 0.4
# 本ツールの入力は顔がほぼ全面を占める切り出し画像で、そのまま 640px に広げると SCRFD の検出範囲（学習時の顔サイズ）を
# 超えて検出できない（実測: 切り出し 49 枚中 43 枚で検出なし）。切り出しをキャンバスの中央に縮めて置き（長辺 = 640 x
# CROP_FRACTION）、周囲をグレーで埋めてから検出する。座標は元の切り出し座標系に戻す。
CROP_FRACTION = 0.5
CANVAS_FILL = 114
INPUT_MEAN = 127.5
INPUT_STD = 128.0
FEAT_STRIDES = (8, 16, 32)  # det_10g: fmc=3, anchors=2, 5 点あり
NUM_ANCHORS = 2


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """insightface.utils.face_align.estimate_norm と同じ（skimage の相似変換の最小二乗推定）。"""
    from skimage import transform as trans
    tform = trans.SimilarityTransform()
    tform.estimate(np.asarray(lmk, np.float32).reshape(5, 2), template_5pts(image_size))
    return tform.params[0:2, :].astype(np.float32)


class SCRFDDetector:
    """det_10g.onnx の推論（insightface SCRFD.detect 相当。入力は BGR）。"""

    def __init__(self, model_file: str):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(model_file, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.batched = len(outputs[0].shape) == 3
        assert len(outputs) == 9, f"det_10g 以外の SCRFD 出力数: {len(outputs)}"
        self._center_cache: dict = {}

    def _forward(self, img: np.ndarray, threshold: float):
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / INPUT_STD, input_size, (INPUT_MEAN,) * 3, swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: np.ascontiguousarray(blob, np.float32)})
        h_in, w_in = blob.shape[2], blob.shape[3]
        fmc = len(FEAT_STRIDES)
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(FEAT_STRIDES):
            scores = net_outs[idx][0] if self.batched else net_outs[idx]
            bbox_preds = (net_outs[idx + fmc][0] if self.batched else net_outs[idx + fmc]) * stride
            kps_preds = (net_outs[idx + fmc * 2][0] if self.batched else net_outs[idx + fmc * 2]) * stride
            height, width = h_in // stride, w_in // stride
            key = (height, width, stride)
            if key not in self._center_cache:
                centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape((-1, 2))
                if NUM_ANCHORS > 1:
                    centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
                self._center_cache[key] = centers
            centers = self._center_cache[key]
            pos = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(centers, bbox_preds)
            kpss = distance2kps(centers, kps_preds).reshape((-1, 5, 2))
            scores_list.append(scores[pos])
            bboxes_list.append(bboxes[pos])
            kpss_list.append(kpss[pos])
        return scores_list, bboxes_list, kpss_list

    def detect(self, img_bgr: np.ndarray, thresh: float = DET_THRESH,
               crop_fraction: float = CROP_FRACTION) -> tuple[np.ndarray, np.ndarray]:
        """-> (dets (N,5)=[x1,y1,x2,y2,score], kpss (N,5,2))。元画像の座標系。
        crop_fraction < 1 なら画像を縮めてキャンバス中央に置く（顔がほぼ全面の切り出し画像用）。"""
        h, w = img_bgr.shape[:2]
        canvas_w, canvas_h = INPUT_SIZE
        scale = min(canvas_w * crop_fraction / w, canvas_h * crop_fraction / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        off_x, off_y = (canvas_w - new_w) // 2, (canvas_h - new_h) // 2
        det_img = np.full((canvas_h, canvas_w, 3), CANVAS_FILL, np.uint8)
        det_img[off_y:off_y + new_h, off_x:off_x + new_w, :] = cv2.resize(img_bgr, (new_w, new_h))
        scores_list, bboxes_list, kpss_list = self._forward(det_img, thresh)
        scores = np.concatenate(scores_list).reshape(-1)
        if scores.size == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        offset = np.array([off_x, off_y], np.float32)
        bboxes = (np.concatenate(bboxes_list) - np.tile(offset, 2)) / scale
        kpss = (np.concatenate(kpss_list) - offset) / scale
        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)[order]
        kpss = kpss[order]
        keep = nms(pre_det, NMS_THRESH)
        return pre_det[keep], kpss[keep]


@register
class InsightFaceSCRFD(FaceAligner):
    name = "InsightFace SCRFD (det_10g)"
    description = ("InsightFace の SCRFD（det_10g.onnx）で 5 点を検出し、insightface の norm_crop と同じ相似変換でアライン。"
                   "多くの学習データ（MS1M-RetinaFace, WebFace）の整列手順に最も近い。初回は buffalo_l を自動取得")

    def __init__(self):
        self.detector: SCRFDDetector | None = None

    def weights_path(self):
        return LANDMARK_DIR / MODEL_FILE

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} のモデル {MODEL_FILE} が見つかりません。画面を開いたときに自動取得します（buffalo_l.zip 約 290MB）。\n"
            "手動で用意する場合は https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip を展開し、"
            f"{MODEL_FILE} を次の場所に配置してください。\n   {self.weights_path()}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if not self.weights_ready():
            ensure_buffalo_file(MODEL_FILE, log)
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.detector is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        self.detector = SCRFDDetector(str(self.weights_path()))
        import onnxruntime as ort
        print(f"[推論環境] 顔ランドマーク: ONNX Runtime {ort.__version__} providers={self.detector.session.get_providers()}"
              f" / モデル: {MODEL_FILE} (SCRFD, 入力 {INPUT_SIZE[0]}x{INPUT_SIZE[1]})", flush=True)

    def detect_main_face(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """クロップ内で最も大きい顔の 5 点（クロップ座標）。見つからなければ None。"""
        self.load()
        dets, kpss = self.detector.detect(face_bgr, DET_THRESH)
        if len(dets) == 0:
            dets, kpss = self.detector.detect(face_bgr, DET_THRESH_RETRY)
        if len(dets) == 0:
            return None
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        return kpss[int(np.argmax(areas))].astype(np.float32)

    def align_with_landmarks(self, face_bgr: np.ndarray, out_size: int) -> AlignResult | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        pts = self.detect_main_face(face_bgr)
        if pts is None or not np.all(np.isfinite(pts)):
            return None
        m = estimate_norm(pts, out_size)
        img = cv2.warpAffine(face_bgr, m, (out_size, out_size), borderValue=0.0)
        out_pts = (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)
        return AlignResult(img, out_pts)

    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        r = self.align_with_landmarks(face_bgr, out_size)
        return None if r is None else r.image
