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

import bisect
import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..camera.distortion import Distortion, Remapper
from .history import load_fisheye_lambdas, write_history
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
FOOT_SOURCE_ANKLE = "ankle"                  # 足元 = 足首キーポイント
FOOT_SOURCE_BBOX_BOTTOM = "bbox_bottom"      # 足元 = 全身 bbox の下辺中央（足首が取れない）
FOOT_SOURCE_EST_PAST = "estimated_past"      # 足元 = それまでの足首の遷移から推測
FOOT_SOURCE_EST_FUTURE = "estimated_future"  # 足元 = それ以降の足首の遷移から後追いで推測（それまでの足首が無い）
FOOT_FIT_POINTS = 5                          # 速度の推定に使う足首の観測点の数
FOOT_MAX_EXTRAPOLATE_SEC = 1.0               # 最後（最初）の観測から進める（戻す）時間の上限


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

    def __init__(self, cap: cv2.VideoCapture, transform: Callable[[np.ndarray], np.ndarray] | None = None):
        super().__init__(daemon=True)
        self.cap = cap
        self.transform = transform  # 各フレームへの前処理（魚眼補正）。デコードスレッドで行う
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
                if self.transform is not None:
                    frame = self.transform(frame)
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


def _ankle_point(kpts) -> list[float] | None:
    """足首（左右）のうち信頼度が KP_CONF_TH 以上のものの平均（フレーム座標）。両方取れなければ None。"""
    k = None if kpts is None else np.asarray(kpts, np.float32)
    if k is not None and k.ndim == 2 and k.shape[0] > IDX_R_ANKLE and k.shape[1] >= 3:
        pts = [k[i, :2] for i in (IDX_L_ANKLE, IDX_R_ANKLE) if k[i, 2] >= KP_CONF_TH]
        if pts:
            x, y = np.mean(pts, axis=0)
            return [round(float(x), 2), round(float(y), 2)]
    return None


def _foot_point(kpts, body_xyxy) -> tuple[list[float], str]:
    """全身骨格の足元の点（フレーム座標）と、その由来（"ankle" / "bbox_bottom"）を返す。
    足首（左右）のうち信頼度が KP_CONF_TH 以上のものの平均。両方取れなければ全身 bbox の下辺中央。"""
    ankle = _ankle_point(kpts)
    if ankle is not None:
        return ankle, FOOT_SOURCE_ANKLE
    x1, y1, x2, y2 = body_xyxy
    return [round((x1 + x2) / 2, 2), float(y2)], FOOT_SOURCE_BBOX_BOTTOM


def _fit_velocity(pts: list[tuple[float, float, float]]) -> tuple[float, float]:
    """(秒, x, y) の並びから最小二乗の速度 (vx, vy)[px/秒]。点が 1 つ・時間幅 0 なら静止（0, 0）。"""
    if len(pts) < 2:
        return 0.0, 0.0
    t = np.array([p[0] for p in pts], np.float64)
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    if denom <= 0:
        return 0.0, 0.0
    return (float((tc * np.array([p[1] for p in pts])).sum() / denom),
            float((tc * np.array([p[2] for p in pts])).sum() / denom))


def estimate_foot_by_motion(obs: list[tuple[float, float, float]], t: float,
                            frame_size: tuple[int, int] | None = None) -> tuple[list[float], str] | None:
    """足首が取れなかった時刻 t の足元を、同じ追跡IDの足首の観測 obs（(秒, x, y)。秒の昇順）の遷移から推測する。
    t より前の観測があれば直近 FOOT_FIT_POINTS 点の速度で最後の観測から進めた位置（"estimated_past"）、
    無ければ t より後の観測の先頭 FOOT_FIT_POINTS 点の速度で最初の観測から戻した位置（"estimated_future"。後追い）。
    進める/戻す時間は FOOT_MAX_EXTRAPOLATE_SEC まで。観測が 1 つも無ければ None。frame_size=(w, h) ならフレーム内に収める。"""
    i = bisect.bisect_left([o[0] for o in obs], t)
    past, future = obs[max(0, i - FOOT_FIT_POINTS):i], obs[i:i + FOOT_FIT_POINTS]
    if past:
        ref, source = past[-1], FOOT_SOURCE_EST_PAST
        dt = min(t - ref[0], FOOT_MAX_EXTRAPOLATE_SEC)
        vx, vy = _fit_velocity(past)
    elif future:
        ref, source = future[0], FOOT_SOURCE_EST_FUTURE
        dt = -min(ref[0] - t, FOOT_MAX_EXTRAPOLATE_SEC)
        vx, vy = _fit_velocity(future)
    else:
        return None
    x, y = ref[1] + vx * dt, ref[2] + vy * dt
    if frame_size is not None:
        x, y = min(max(x, 0.0), frame_size[0] - 1.0), min(max(y, 0.0), frame_size[1] - 1.0)
    return [round(float(x), 2), round(float(y), 2)], source


def _apply_foot_estimates(records: dict[int, list[dict]], foot_obs: dict[int, list[tuple[float, float, float]]],
                          frame_size: tuple[int, int] | None) -> None:
    """足首が取れずに bbox 下辺で出した記録の foot_xy を、足首の観測の遷移から推測した位置に置き換える（後追い）。
    X は推測値ではなく、その記録の顔 bbox（頭）の左右の X の中央、Y は遷移からの推測値。"""
    for tid, imgs in records.items():
        obs = foot_obs.get(tid, [])
        for im in imgs:
            if im["foot_source"] != FOOT_SOURCE_BBOX_BOTTOM:
                continue
            est = estimate_foot_by_motion(obs, im["time_sec"], frame_size)
            if est is not None:
                (_, y), source = est
                fx1, _, fx2, _ = im["face_bbox_xyxy"]
                im["foot_xy"], im["foot_source"] = [round((fx1 + fx2) / 2, 2), y], source


def extract_video(
    video_path: Path,
    out_dir: Path,
    pose_detector: PoseDetector,
    face_detector: FaceDetector,
    interval_sec: float,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    location: str = "",
    estimate_foot: bool = False,
    fisheye_lambda: float | None = None,
) -> ExtractResult:
    """out_dir = 出力先/日時/カメラID/映像名(拡張子なし)。直下に追跡ID毎のフォルダを作る。
    estimate_foot=True: 足首が取れない記録の foot_xy を、足首の遷移から推測した位置にする（estimate_foot_by_motion）。
    fisheye_lambda: 指定すると、各フレームを魚眼補正（レンズ歪みの補正。垂直補正はしない）してから処理する。
    画像・json の座標はすべて補正後のフレームのもの（json に fisheye_corrected / fisheye_lambda を記録）。"""
    result = ExtractResult(video_path=video_path, out_dir=out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"映像を開けません: {video_path}")
    transform = None
    if fisheye_lambda is not None:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        transform = Remapper(Distortion(fisheye_lambda, size)).apply
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_dt = parse_capture_datetime(video_path.name)
    tracker = PersonTracker(fps)
    last_out: dict[int, float] = {}   # 追跡ID -> 採用基準を通った顔を最後に出力した秒（出力対象の判定はこれだけで行う）
    last_rej: dict[int, float] = {}   # 追跡ID -> 採用基準に外れた顔を最後に出力した秒
    records: dict[int, list[dict]] = {}
    foot_obs: dict[int, list[tuple[float, float, float]]] = {}  # 追跡ID -> 毎フレームの足首の観測（秒, x, y）。推測用
    frame_size: tuple[int, int] | None = None
    decoder = _Decoder(cap, transform)
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
                if estimate_foot:  # 出力しないフレームも含めて足首の遷移を記録する
                    frame_size = (frame.shape[1], frame.shape[0])
                    for p in persons:
                        ankle = _ankle_point(p.kpts)
                        if ankle is not None:
                            foot_obs.setdefault(p.track_id, []).append((cur_sec, ankle[0], ankle[1]))
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
                        "foot_source": foot_source,                           # "ankle"=足首 / "bbox_bottom"=全身bbox下辺中央 / "estimated_past"・"estimated_future"=足首の遷移から推測
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
        if estimate_foot:
            _apply_foot_estimates(records, foot_obs, frame_size)
        for tid, imgs in records.items():
            data = {"track_id": tid, "video_file": video_path.name,
                    "location": location, "fisheye_corrected": fisheye_lambda is not None,
                    "fisheye_lambda": fisheye_lambda, "images": imgs}
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
    estimate_foot: bool = False,
    fisheye_correct: bool = False,
    camera_dir: Path | str | None = None,
    inputs: dict | None = None,
) -> tuple[list[ExtractResult], Path]:
    """複数映像の抽出（GUI 非依存。ui/worker.py と cli.py の両方から呼ぶ）。
    progress は (映像index, 映像数, フレーム, 総フレーム)。戻り値は (結果一覧, 出力先フォルダ)。
    fisheye_correct=True: camera_dir の <カメラID>.json の歪み補正（λ）で各フレームを魚眼補正してから処理する。
    設定ファイルが無い・魚眼補正の情報が無いカメラがあれば、処理を始める前に FisheyeConfigError で終了する。
    inputs: 画面で入力した内容。全映像の処理後に 抽出履歴.json として日時フォルダ直下へ出力する。"""
    from .face_detector import prepare_face_detector
    from .pose_model import prepare_pose_detector

    lambdas: dict[str, float] = {}
    if fisheye_correct:
        lambdas = load_fisheye_lambdas(camera_dir, [v.camera_id for v in videos])
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
            is_cancelled=is_cancelled, location=v.location, estimate_foot=estimate_foot,
            fisheye_lambda=lambdas[v.camera_id] if fisheye_correct else None,
        )
        results.append(res)
        if res.cancelled:
            break
    history = {
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        **(inputs or {}),
        "interval_sec": interval_sec,
        "estimate_foot": estimate_foot,
        "fisheye_correct": fisheye_correct,
        "camera_dir": str(camera_dir) if fisheye_correct and camera_dir else "",
        "fisheye_lambdas": lambdas,        # カメラID -> 歪み係数 λ（魚眼補正したときだけ）
        "cancelled": any(r.cancelled for r in results),
        "videos": [{"camera_id": v.camera_id, "location": v.location, "file_name": v.file_name, "path": str(v.path),
                    "tracks": r.track_count, "images": r.image_count, "cancelled": r.cancelled}
                   for v, r in zip(videos, results)],
    }
    write_history(run_dir, history)
    return results, run_dir
