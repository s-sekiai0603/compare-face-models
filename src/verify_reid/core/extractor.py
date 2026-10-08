"""映像1本からの外見画像抽出（参照元「デバッグ画像抽出」相当）。"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2

from .body_align import align_person_crop
from .botsort_tracker import PersonTracker
from .pose_detector import BATCH_SIZE, PoseDetector
from .video_scanner import parse_capture_datetime

MIN_CROP_SIDE = 48
JPEG_QUALITY = 90
DECODE_PREFETCH = 4 * BATCH_SIZE  # デコード済みフレームの先読み上限
WRITE_QUEUE_SIZE = 256            # 画像保存待ちの上限
WRITER_THREADS = 2


@dataclass
class ExtractResult:
    video_path: Path
    out_dir: Path
    track_count: int = 0
    image_count: int = 0
    cancelled: bool = False


def make_timestamp(start: datetime | None, frame_index: int, fps: float) -> str:
    """YYYYMMDD_HHMMSS_mmm。撮影日時が不明ならフレーム番号ベース。"""
    if start is None:
        return f"frame{frame_index:08d}_000"
    t = start + timedelta(seconds=frame_index / fps)
    return t.strftime("%Y%m%d_%H%M%S_") + f"{t.microsecond // 1000:03d}"


def should_output(last_sec: float | None, cur_sec: float, interval_sec: float) -> bool:
    return last_sec is None or (cur_sec - last_sec) >= interval_sec


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
    """人物クロップの回転補正とJPEG保存を行う別スレッド群。"""

    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self.error: Exception | None = None
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(WRITER_THREADS)]
        for t in self.threads:
            t.start()

    def submit(self, tid_dir: Path, f_body: str, f_aligned: str, crop, kpts) -> None:
        self.q.put((tid_dir, f_body, f_aligned, crop, kpts))

    def _run(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        while True:
            job = self.q.get()
            if job is None:
                return
            tid_dir, f_body, f_aligned, crop, kpts = job
            try:
                try:
                    aligned = align_person_crop(crop, kpts)
                except Exception:
                    aligned = crop
                tid_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tid_dir / f_body), crop, params)
                cv2.imwrite(str(tid_dir / f_aligned), aligned, params)
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


def extract_video(
    video_path: Path,
    out_dir: Path,
    detector: PoseDetector,
    interval_sec: float,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    location: str = "",
) -> ExtractResult:
    """out_dir = 出力先/日時/カメラID/映像名(拡張子なし)。直下に追跡ID毎のフォルダを作る。

    デコード先読み → バッチ推論 → 追跡(フレーム順に逐次) → 画像保存(別スレッド) の
    パイプライン。追跡はフレーム順・1フレームずつ更新するため、結果は逐次処理と変わらない。
    """
    result = ExtractResult(video_path=video_path, out_dir=out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"映像を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_dt = parse_capture_datetime(video_path.name)
    tracker = PersonTracker(fps)
    last_out: dict[int, float] = {}
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
            detections = detector.detect_batch(frames)  # バッチ推論
            for (fi, frame), det in zip(batch, detections):  # 追跡はフレーム順に逐次
                persons = tracker.update(det, frame)
                cur_sec = fi / fps
                h, w = frame.shape[:2]
                for p in persons:
                    if not should_output(last_out.get(p.track_id), cur_sec, interval_sec):
                        continue
                    x1, y1, x2, y2 = (int(v) for v in p.bbox_xyxy)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1 or min(x2 - x1, y2 - y1) < MIN_CROP_SIDE:
                        continue
                    ts = make_timestamp(start_dt, fi, fps)
                    f_body, f_aligned = f"{ts}_body.jpg", f"{ts}_body_aligned.jpg"
                    writer.submit(out_dir / str(p.track_id), f_body, f_aligned,
                                  frame[y1:y2, x1:x2], p.kpts)
                    last_out[p.track_id] = cur_sec
                    records.setdefault(p.track_id, []).append({
                        "file_body": f_body,
                        "file_body_aligned": f_aligned,
                        "frame_index": fi,
                        "time_sec": round(cur_sec, 3),
                        "timestamp": ts,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "person_conf": round(p.score, 4),
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
            (out_dir / str(tid) / f"{tid}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if decoder.error:
        raise decoder.error
    if writer.error:
        raise writer.error
    result.track_count = len(records)
    if progress:
        progress(total, total)
    return result
