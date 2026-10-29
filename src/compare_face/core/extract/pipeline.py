"""映像1本からの顔画像抽出。

デコード先読み → 人物(YOLOv8-pose)と顔(YOLOv8-Face)のバッチ推論 → 追跡(BoT-SORT, フレーム順に逐次)
→ 顔と人物の紐付け（顔が付かない人物は全身クロップで顔を再検出） → 顔ゲート/クロップ
→ 画像保存(別スレッド) のパイプライン。追跡はフレーム順に1枚ずつ更新するため、結果は逐次処理と同じ。

出力は追跡IDごとに interval_sec 秒間隔。顔ゲートを通った顔がある最初のフレームで出力する
（premises-security のデバッグ画像抽出は全身を間隔出力しその瞬間に顔があれば顔も出すが、
本ツールは顔が主役のため「顔が出せたフレーム」を基準に間隔を数える）。

採用基準（顔ゲート・品質ガード）に外れた顔も保存する（JSON の gate_ok=false）。採用される顔の出力タイミングと処理量を
変えないため、基準外の顔は採用の間隔内では出さず、採用できる顔が無い間だけ基準外の間隔で出す（decide_output）。
使う・使わないはレポート時に決める。
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .tracker import PersonTracker, TrackedPerson
from .face_crop import KP_CONF_TH, attach_faces, crop_guard_reason, evaluate_face_gate, make_face_crop, quality_score
from .face_detector import FaceDetector
from ..common.imageio import imwrite
from .pose_detector import BATCH_SIZE, PoseDetector
from .validity import face_box_mismatch_reason, foreign_skeleton_reason
from .video_scanner import VideoInfo, parse_capture_datetime

JPEG_QUALITY = 90
DECODE_PREFETCH = 4 * BATCH_SIZE  # デコード済みフレームの先読み上限
WRITE_QUEUE_SIZE = 256            # 画像保存待ちの上限
WRITER_THREADS = 2
FACE_SOURCE_FRAME = "frame"       # フレーム全体の顔検出で見つかった顔
FACE_SOURCE_BODY = "body_crop"    # 全身クロップでの再検出で見つかった顔


@dataclass
class ExtractResult:
    video_path: Path
    out_dir: Path
    track_count: int = 0
    image_count: int = 0    # 保存した顔画像の数（採用基準外を含む）
    rejected_count: int = 0  # 採用基準（顔ゲート・品質ガード）に外れたが保存した顔画像の数
    invalid_count: int = 0
    cancelled: bool = False


def make_timestamp(start: datetime | None, frame_index: int, fps: float) -> str:
    """YYYYMMDD_HHMMSS_mmm。撮影日時が不明ならフレーム番号ベース。"""
    if start is None:
        return f"frame{frame_index:08d}_000"
    t = start + timedelta(seconds=frame_index / fps)
    return t.strftime("%Y%m%d_%H%M%S_") + f"{t.microsecond // 1000:03d}"


def should_output(last_sec: float | None, cur_sec: float, interval_sec: float) -> bool:
    return last_sec is None or (cur_sec - last_sec) >= interval_sec


OUTPUT_ACCEPTED = "accepted"   # 採用基準を通った顔（premises-security が出力する顔）
OUTPUT_REJECTED = "rejected"   # 採用基準に外れた顔（本ツールだけが保存する。レポート時に使うか決める）


def decide_output(last_accepted: float | None, last_rejected: float | None, cur_sec: float,
                  interval_sec: float, gate_ok: bool) -> str | None:
    """この顔を出力するか。採用基準を通った顔は従来どおり採用の間隔だけで判定する（基準外の出力は影響しない）。
    基準外の顔は、採用の間隔内（採用できる顔を出した直後）は出さず、採用できる顔が無い間だけ基準外の間隔で出す。
    これで採用される顔の出力タイミングと 1 フレームあたりの処理量は、基準外の顔を保存しない場合と同じになる。
    戻り値は OUTPUT_ACCEPTED / OUTPUT_REJECTED / None（出力しない）。"""
    if not should_output(last_accepted, cur_sec, interval_sec):
        return None
    if gate_ok:
        return OUTPUT_ACCEPTED
    return OUTPUT_REJECTED if should_output(last_rejected, cur_sec, interval_sec) else None


class _Decoder(threading.Thread):
    """映像デコード専用スレッド。(frame_index, frame) をキューへ先読みする。"""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self.q: queue.Queue = queue.Queue(maxsize=DECODE_PREFETCH)
        self.stop_event = threading.Event()
        self.error: Exception | None = None

    def _put(self, item) -> bool:
        while not self.stop_event.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run(self) -> None:
        try:
            fi = 0
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                if not self._put((fi, frame)):
                    return
                fi += 1
        except Exception as e:
            self.error = e
        finally:
            self._put(None)  # 終端


class _ImageWriter:
    """JPEG 保存を行う別スレッド群。"""

    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self.error: Exception | None = None
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(WRITER_THREADS)]
        for t in self.threads:
            t.start()

    def submit(self, path: Path, img: np.ndarray) -> None:
        self.q.put((path, img))

    def _run(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        while True:
            job = self.q.get()
            if job is None:
                return
            path, img = job
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not imwrite(path, img, params):  # 日本語パス対応
                    raise OSError(f"画像を保存できません: {path}")
            except Exception as e:
                self.error = self.error or e

    def close(self) -> None:
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()


def _read_batch(decoder: _Decoder) -> tuple[list[tuple[int, object]], bool]:
    """先読みキューから最大 BATCH_SIZE 枚取り出す。戻り値の bool は終端到達。"""
    batch = []
    while len(batch) < BATCH_SIZE:
        item = decoder.q.get()
        if item is None:
            return batch, True
        batch.append(item)
    return batch, False


def _clip_body(bbox, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _retry_faces_on_body(face_detector: FaceDetector, frame: np.ndarray,
                         persons: list[TrackedPerson]) -> list[tuple[np.ndarray, float] | None]:
    """顔が付かなかった人物の全身クロップで顔を再検出し、最もスコアの高い顔をフレーム座標で返す
    （フルフレームの 640px 推論では小さすぎて検出できない遠くの顔を拾う。premises-security の
    _retry_faces_on_body_crops と同じ）。"""
    h, w = frame.shape[:2]
    out: list[tuple[np.ndarray, float] | None] = [None] * len(persons)
    crops, offs = [], []
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = _clip_body(p.bbox_xyxy, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(frame[y1:y2, x1:x2])
        offs.append((i, x1, y1))
    if not crops:
        return out
    for (i, ox, oy), det in zip(offs, face_detector.detect_batch(crops)):
        if len(det.boxes) == 0:
            continue
        j = int(np.argmax(det.scores))
        out[i] = (det.boxes[j] + np.array([ox, oy, ox, oy], np.float32), float(det.scores[j]))
    return out


def _round_or_none(v, nd: int):
    return None if v is None else round(float(v), nd)


IDX_L_ANKLE, IDX_R_ANKLE = 15, 16


def _foot_point(kpts, body_xyxy) -> tuple[list[float], str]:
    """全身骨格の足元の点（フレーム座標）と、その由来（"ankle" / "bbox_bottom"）を返す。
    足首（左右）のうち信頼度が KP_CONF_TH 以上のものの平均。両方取れなければ全身 bbox の下辺中央。"""
    k = None if kpts is None else np.asarray(kpts, np.float32)
    if k is not None and k.ndim == 2 and k.shape[0] > IDX_R_ANKLE and k.shape[1] >= 3:
        pts = [k[i, :2] for i in (IDX_L_ANKLE, IDX_R_ANKLE) if k[i, 2] >= KP_CONF_TH]
        if pts:
            x, y = np.mean(pts, axis=0)
            return [round(float(x), 2), round(float(y), 2)], "ankle"
    x1, y1, x2, y2 = body_xyxy
    return [round((x1 + x2) / 2, 2), float(y2)], "bbox_bottom"


def extract_video(
    video_path: Path,
    out_dir: Path,
    pose_detector: PoseDetector,
    face_detector: FaceDetector,
    interval_sec: float,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    location: str = "",
) -> ExtractResult:
    """out_dir = 出力先/日時/カメラID/映像名(拡張子なし)。直下に追跡ID毎のフォルダを作る。"""
    result = ExtractResult(video_path=video_path, out_dir=out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"映像を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_dt = parse_capture_datetime(video_path.name)
    tracker = PersonTracker(fps)
    last_out: dict[int, float] = {}   # 追跡ID -> 採用基準を通った顔を最後に出力した秒（出力対象の判定はこれだけで行う）
    last_rej: dict[int, float] = {}   # 追跡ID -> 採用基準に外れた顔を最後に出力した秒
    records: dict[int, list[dict]] = {}
    decoder = _Decoder(cap)
    writer = _ImageWriter()
    decoder.start()

    try:
        done = False
        while not done:
            if is_cancelled and is_cancelled():
                result.cancelled = True
                break
            batch, done = _read_batch(decoder)
            if not batch:
                break
            frames = [f for _, f in batch]
            pose_dets = pose_detector.detect_batch(frames)  # バッチ推論（人物+骨格）
            face_dets = face_detector.detect_batch(frames)  # バッチ推論（顔）
            for (fi, frame), pdet, fdet in zip(batch, pose_dets, face_dets):  # 追跡はフレーム順に逐次
                persons = tracker.update(pdet, frame)
                cur_sec = fi / fps
                due = [i for i, p in enumerate(persons)
                       if should_output(last_out.get(p.track_id), cur_sec, interval_sec)]
                if not due:
                    continue
                assigned = attach_faces([p.bbox_xyxy for p in persons], fdet.boxes, fdet.scores)
                faces = {i: (fdet.boxes[a], float(fdet.scores[a]), FACE_SOURCE_FRAME)
                         for i, a in enumerate(assigned) if a is not None}
                need = [i for i in due if i not in faces]
                if need:
                    for i, r in zip(need, _retry_faces_on_body(face_detector, frame, [persons[i] for i in need])):
                        if r is not None:
                            faces[i] = (r[0], r[1], FACE_SOURCE_BODY)
                h, w = frame.shape[:2]
                for i in due:
                    if i not in faces:
                        continue
                    p = persons[i]
                    fbox, fconf, src = faces[i]
                    gate = evaluate_face_gate(fbox, p.kpts)
                    fc = make_face_crop(frame, fbox) if gate.ok else None  # 品質ガードは顔ゲートを通った顔だけ判定すればよい
                    gate_ok = gate.ok and fc.guard_ok   # premises-security ならこの顔を出力するか
                    kind = decide_output(last_out.get(p.track_id), last_rej.get(p.track_id), cur_sec, interval_sec, gate_ok)
                    if kind is None:
                        continue
                    if fc is None:
                        fc = make_face_crop(frame, fbox)  # 基準外の顔は出力するときだけ切り出す（処理量を従来と同じに保つ）
                    if fc.image is None:
                        continue
                    ts = make_timestamp(start_dt, fi, fps)
                    tid_dir = out_dir / str(p.track_id)
                    crop_xyxy = tuple(int(v) for v in fc.inflated)
                    reason = (foreign_skeleton_reason(p.bbox_xyxy, pdet.all_boxes, pdet.all_kpts, crop_xyxy)
                              or face_box_mismatch_reason(p.kpts, fbox))
                    f_face = f"{ts}_face.jpg"
                    writer.submit(tid_dir / f_face, fc.image)
                    bx1, by1, bx2, by2 = _clip_body(p.bbox_xyxy, w, h)
                    f_body = None
                    if bx2 > bx1 and by2 > by1:
                        f_body = f"{ts}_body.jpg"
                        writer.submit(tid_dir / f_body, frame[by1:by2, bx1:bx2])
                    foot_xy, foot_source = _foot_point(p.kpts, (bx1, by1, bx2, by2))
                    (last_out if kind == OUTPUT_ACCEPTED else last_rej)[p.track_id] = cur_sec
                    records.setdefault(p.track_id, []).append({
                        "file_face": f_face,
                        "file_body": f_body,
                        "frame_index": fi,
                        "time_sec": round(cur_sec, 3),
                        "timestamp": ts,
                        "bbox_xyxy": [bx1, by1, bx2, by2],
                        "person_conf": round(p.score, 4),
                        "face_bbox_xyxy": [round(float(v), 1) for v in fbox],
                        "face_crop_xyxy": list(crop_xyxy),
                        "face_conf": round(fconf, 4),
                        "face_source": src,
                        "face_width": round(gate.face_width, 1),
                        "face_height": round(float(fbox[3] - fbox[1]), 1),
                        "foot_xy": foot_xy,                                   # 足元の点 [x, y]（フレーム座標）
                        "foot_source": foot_source,                           # "ankle"=足首キーポイント / "bbox_bottom"=全身bbox下辺中央
                        "gate_ok": gate_ok,                                   # 採用基準（顔ゲート + 品質ガード）を通ったか
                        "gate_reason": gate.reason or crop_guard_reason(fc.min_side),  # 外れた理由（通れば空）
                        "face_visible": gate.face_visible,                   # 骨格ルール（鼻+片目 / 下向きでない）を通ったか
                        "crop_min_side": fc.min_side,                        # 切り出し box の短辺(px)。品質ガードの判定値
                        "eye_ratio": _round_or_none(gate.eye_ratio, 4),
                        "frontal": gate.frontal,
                        "nose_conf": _round_or_none(gate.nose_conf, 4),
                        "l_eye_conf": _round_or_none(gate.l_eye_conf, 4),
                        "r_eye_conf": _round_or_none(gate.r_eye_conf, 4),
                        "quality": round(quality_score(fconf, gate.face_width), 4),
                        "invalid": reason is not None,
                        "invalid_reason": reason or "",
                        "keypoints": [{"x": round(float(x), 2), "y": round(float(y), 2),
                                       "conf": round(float(c), 4)} for x, y, c in p.kpts],
                    })
                    result.image_count += 1
            if progress:
                progress(batch[-1][0], total)
    finally:
        decoder.stop_event.set()
        try:  # 満杯で詰まっているデコーダを解放する
            while True:
                decoder.q.get_nowait()
        except queue.Empty:
            pass
        decoder.join(timeout=5)
        cap.release()
        writer.close()  # 保存待ちを全て書き終えてからJSONを出力する
        for tid, imgs in records.items():
            data = {"track_id": tid, "video_file": video_path.name,
                    "location": location, "images": imgs}
            (out_dir / str(tid)).mkdir(parents=True, exist_ok=True)
            (out_dir / str(tid) / f"{tid}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if decoder.error:
        raise decoder.error
    if writer.error:
        raise writer.error
    result.track_count = len(records)
    result.rejected_count = sum(1 for imgs in records.values() for im in imgs if not im["gate_ok"])
    result.invalid_count = sum(1 for imgs in records.values() for im in imgs if im["invalid"])
    if progress:
        progress(total, total)
    return result


def run_extraction(
    videos: list[VideoInfo],
    output_root: Path,
    interval_sec: float,
    status: Callable[[str], None] = print,
    progress: Callable[[int, int, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[ExtractResult], Path]:
    """複数映像の抽出（GUI 非依存。ui/worker.py と cli.py の両方から呼ぶ）。
    progress は (映像index, 映像数, フレーム, 総フレーム)。戻り値は (結果一覧, 出力先フォルダ)。"""
    from .face_detector import prepare_face_detector
    from .pose_model import prepare_pose_detector

    status("モデルを準備しています...")
    pose = prepare_pose_detector(log=status)
    face = prepare_face_detector(log=status)
    run_dir = output_root / datetime.now().strftime("%Y%m%d%H%M%S")
    results: list[ExtractResult] = []
    for i, v in enumerate(videos):
        status(f"処理中 ({i + 1}/{len(videos)}): {v.file_name}")
        out_dir = run_dir / v.camera_id / v.path.stem
        res = extract_video(
            v.path, out_dir, pose, face, interval_sec,
            progress=(lambda f, t, i=i: progress(i, len(videos), f, t)) if progress else None,
            is_cancelled=is_cancelled, location=v.location,
        )
        results.append(res)
        if res.cancelled:
            break
    return results, run_dir
