"""射影変換設定抽出画面。

カメラ画像上の「現実では長方形になる 4 点」と長方形の 2 辺の長さから、カメラの設置座標（画像座標系。画像の外になり得る）と
画像 → 地面（m）の射影変換を求め、距離シミュレーションと設定ファイル（<画像名>.json）の保存を行う（core/camera/projective.py）。
最後に選んだ画像・出力先は camera_projection_screen_setting.json に保存し、次回起動時に復元する。"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ..core.camera.projective import (
    CameraProjection, ProjectionError, calibrate_focal, clip_line_to_rect, compute_projection, load_config, resolve_sides, save_config,
)
from ..core.camera.distortion import MIN_LINE_POINTS, Distortion, estimate_lambda, lambda_range
from ..core.camera.vertical import (
    MIN_VERTICAL_LINES, VerticalRectification, line_tilt_deg, vertical_vanishing_point,
)
from ..core.common.imageio import imread
from ..core.common.settings import (
    CAMERA_PROJECTION_IMAGE_DIR, load_camera_projection_settings, save_camera_projection_settings,
)
from .widgets import add_path_row, fixed_label

IMAGE_FILTER = "画像ファイル (*.png *.jpg *.jpeg *.bmp);;すべてのファイル (*)"
POINT_COLOR = QColor(255, 200, 0)
CAMERA_COLOR = QColor(255, 60, 60)
SIM_COLOR = QColor(0, 220, 255)
REF_COLOR = QColor(120, 255, 120)
HORIZON_COLOR = QColor(255, 90, 255)
DIST_COLOR = QColor(0, 255, 170)  # 歪み補正線
MEMO_COLOR = QColor(255, 105, 180)  # 一時的なメモ線（保存しない）
VLINE_COLOR = QColor(240, 240, 240)  # 垂直補正に使う建物の垂直線
VP_COLORS = (QColor(255, 150, 0), QColor(110, 170, 255))  # 消失点 1（辺 1-2 と 4-3）/ 消失点 2（辺 2-3 と 1-4）
VP_DRAW_LIMIT = 4.0  # 画像の長辺のこの倍率より遠い消失点は描かない（補線はその距離で打ち切る）
DRAG_THRESHOLD_PX = 5
POINT_HIT_PX = 12  # 点をつかめる範囲（画面上のピクセル）
MAX_SIDE_M = 10000.0
MIN_SCALE = 0.001
MAX_SCALE = 50.0


def _marker(scene: QGraphicsScene, pos: QPointF, color: QColor, radius: float, text: str = "") -> None:
    """画面上の大きさが変わらない丸印と文字（拡大縮小の影響を受けない）。"""
    dot = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
    dot.setBrush(QBrush(color))
    dot.setPen(QPen(Qt.black, 1))
    dot.setFlag(QGraphicsItem.ItemIgnoresTransformations)
    dot.setPos(pos)
    dot.setZValue(10)
    scene.addItem(dot)
    if text:
        label = QGraphicsSimpleTextItem(text)
        label.setBrush(QBrush(color))
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        font = label.font()
        font.setBold(True)
        font.setPointSize(11)
        label.setFont(font)
        label.setPos(pos + QPointF(0, 0))
        label.setZValue(11)
        scene.addItem(label)


class ImageView(QGraphicsView):
    """画像 + 4 点 + カメラ設置座標 + シミュレーション点を表示する。ホイールで拡大縮小、ドラッグで移動、クリックで座標通知。"""
    image_clicked = Signal(float, float)
    point_moved = Signal(int, float, float)  # ドラッグ中の 4 点（番号, 画像座標）
    vline_moved = Signal(int, int, float, float)  # ドラッグ中の垂直線の端点（線の番号, 0=始点/1=終点, 画像座標）
    memo_moved = Signal(int, int, float, float)  # ドラッグ中のメモ線の端点（線の番号, 0=始点/1=終点, 画像座標）

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(60, 60, 60)))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumHeight(360)
        self._image_rect = QRectF()
        self._points: list[tuple[float, float]] = []
        self._camera: tuple[float, float] | None = None
        self._sim: tuple[tuple[float, float], str] | None = None
        self._ref: tuple[float, float] | None = None
        self._horizon = None  # 画像内に切り取った地平線の線分 ((x, y), (x, y))
        self._vanishing = None  # 消失点 ((x, y), (x, y))（補線は _points の辺から作る）
        self._dist_lines: list[list[tuple[float, float]]] = []  # 歪み補正線（表示中の画像の座標）
        self._memo_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []  # メモ線（表示中の画像の座標）
        self._memo_pending: tuple[float, float] | None = None  # メモ線の始点だけ指定した状態
        self._vlines: list[tuple[tuple[float, float], tuple[float, float]]] = []  # 垂直線（表示中の画像の座標）
        self._vpending: tuple[float, float] | None = None
        self._show_points = True  # 長方形の頂点（番号付きマーカー）を描くか。辺・塗り・消失点の補線などは頂点の表示に関係なく描く
        self._press_pos = None
        self._dragged = False
        self._drag_idx: int | None = None
        self._drag_memo: tuple[int, int] | None = None
        self._drag_vline: tuple[int, int] | None = None
        self.vline_drag_enabled = True  # 垂直補正の適用中・他の指定モード中は垂直線の端点を動かさない
        self.memo_drag_enabled = True  # 距離シミュレーション・補正点指定・歪み補正線指定中はメモ線の端点を動かさない
        self.point_drag_enabled = True  # 距離シミュレーション・補正点指定中は点を動かさない
        self.viewport().setMouseTracking(True)

    # ---- 表示内容 ----
    def set_image(self, image: QImage | None) -> None:
        self._points, self._camera, self._sim, self._ref = [], None, None, None
        self._horizon, self._vanishing = None, None
        self._dist_lines = []
        self._memo_lines, self._memo_pending = [], None
        self._vlines, self._vpending = [], None
        self._image_rect = QRectF(0, 0, image.width(), image.height()) if image is not None else QRectF()
        self._rebuild(image)
        self.fit_all()

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = list(points)
        self._rebuild()

    def set_camera(self, xy: tuple[float, float] | None) -> None:
        self._camera = xy
        self._rebuild()

    def set_sim(self, sim: tuple[tuple[float, float], str] | None) -> None:
        self._sim = sim
        self._rebuild()

    def set_guides(self, horizon, vanishing) -> None:
        """地平線の線分（画像内に切り取り済み）と消失点 2 つ。どちらも None なら消す。"""
        self._horizon = horizon
        self._vanishing = vanishing
        self._rebuild()

    def _vp_draw_info(self):
        """描く消失点ごとの (色, 消失点, 補線の線分 2 本, 描くか)。消失点が遠すぎるときは補線だけを打ち切って描く。"""
        if self._vanishing is None or len(self._points) != 4:
            return []
        p = self._points
        edges = (((p[0], p[1]), (p[3], p[2])), ((p[1], p[2]), (p[0], p[3])))
        center = self._image_rect.center()
        limit = VP_DRAW_LIMIT * max(self._image_rect.width(), self._image_rect.height())
        info = []
        for color, vp, pair in zip(VP_COLORS, self._vanishing, edges):
            near = ((vp[0] - center.x()) ** 2 + (vp[1] - center.y()) ** 2) ** 0.5 <= limit
            segments = []
            for a, b in pair:
                da = (a[0] - vp[0]) ** 2 + (a[1] - vp[1]) ** 2
                db = (b[0] - vp[0]) ** 2 + (b[1] - vp[1]) ** 2
                far_pt, near_pt = (a, b) if da > db else (b, a)  # 消失点から遠い方の端点から、消失点の方へ延長する
                if near:
                    end = vp
                else:
                    dx, dy = near_pt[0] - far_pt[0], near_pt[1] - far_pt[1]
                    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
                    end = (far_pt[0] + dx / norm * limit, far_pt[1] + dy / norm * limit)
                segments.append((far_pt, end))
            info.append((color, vp, segments, near))
        return info

    def set_background_image(self, image: QImage) -> None:
        """同じ大きさの画像に差し替える（4 点などの重ね描きと表示倍率はそのまま）。"""
        self._rebuild(image)

    def set_dist_lines(self, lines) -> None:
        self._dist_lines = [list(line) for line in lines]
        self._rebuild()

    def set_memo(self, lines, pending=None) -> None:
        self._memo_lines = [(tuple(a), tuple(b)) for a, b in lines]
        self._memo_pending = pending
        self._rebuild()

    def set_vlines(self, lines, pending=None) -> None:
        self._vlines = [(tuple(a), tuple(b)) for a, b in lines]
        self._vpending = pending
        self._rebuild()

    def set_ref(self, xy: tuple[float, float] | None) -> None:
        self._ref = xy
        self._rebuild()

    def _rebuild(self, image: QImage | None = None) -> None:
        pixmap_item = None
        if image is not None:
            self._scene.clear()
            pixmap_item = self._scene.addPixmap(QPixmap.fromImage(image))
        else:
            for item in list(self._scene.items()):
                if item.zValue() > 0:
                    self._scene.removeItem(item)
        if self._image_rect.isEmpty():
            return
        if pixmap_item is not None:
            border = self._scene.addRect(self._image_rect, QPen(QColor(200, 200, 200), 0))
            border.setZValue(0)  # 0 以下は overlay 再構築で消さない
        s = self._scene
        if len(self._points) >= 2:
            poly = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in self._points]))
            poly.setPen(QPen(POINT_COLOR, 0))
            poly.setBrush(QBrush(QColor(255, 200, 0, 40)))
            poly.setZValue(5)
            s.addItem(poly)
        for i, (x, y) in enumerate(self._points if self._show_points else []):
            _marker(s, QPointF(x, y), POINT_COLOR, 5, f" {i + 1}")
        if self._camera is not None:
            cx, cy = self._camera
            _marker(s, QPointF(cx, cy), CAMERA_COLOR, 8, " カメラ設置座標")
            center = self._image_rect.center()
            line = QGraphicsLineItem(cx, cy, center.x(), center.y())  # 画像外でも位置関係が分かるよう画像中心と結ぶ
            pen = QPen(CAMERA_COLOR, 0, Qt.DashLine)
            line.setPen(pen)
            line.setZValue(4)
            s.addItem(line)
        if self._horizon is not None:
            (x1, y1), (x2, y2) = self._horizon
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(HORIZON_COLOR, 0, Qt.DashLine))  # cosmetic（太さ 0）のまま点線
            line.setZValue(3)
            s.addItem(line)
            label = QGraphicsSimpleTextItem("地平線")
            label.setBrush(QBrush(HORIZON_COLOR))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            label.setPos(QPointF(min(x1, x2), min(y1, y2) if abs(y1 - y2) < 1 else (y1 if x1 < x2 else y2)))
            label.setZValue(3)
            s.addItem(label)
        for i, (color, vp, segments, near) in enumerate(self._vp_draw_info()):
            for (x1, y1), (x2, y2) in segments:  # 消失点を求めるための補線（長方形の辺の延長）
                aux = QGraphicsLineItem(x1, y1, x2, y2)
                aux.setPen(QPen(color, 0, Qt.DotLine))
                aux.setZValue(3)
                s.addItem(aux)
            if near:
                _marker(s, QPointF(*vp), color, 6, f" 消失点{i + 1}")
        for line in self._dist_lines:  # 歪み補正線（補正後の画像ではまっすぐになるはず）
            if len(line) >= 2:
                path = QPainterPath(QPointF(*line[0]))
                for x, y in line[1:]:
                    path.lineTo(x, y)
                item = QGraphicsPathItem(path)
                item.setPen(QPen(DIST_COLOR, 0))
                item.setZValue(4)
                s.addItem(item)
            for x, y in line:
                _marker(s, QPointF(x, y), DIST_COLOR, 3)
        for (x1, y1), (x2, y2) in self._vlines:
            item = QGraphicsLineItem(x1, y1, x2, y2)
            item.setPen(QPen(VLINE_COLOR, 0))
            item.setZValue(6)
            s.addItem(item)
            _marker(s, QPointF(x1, y1), VLINE_COLOR, 3)
            _marker(s, QPointF(x2, y2), VLINE_COLOR, 3)
        if self._vpending is not None:
            _marker(s, QPointF(*self._vpending), VLINE_COLOR, 3)
        for (x1, y1), (x2, y2) in self._memo_lines:
            item = QGraphicsLineItem(x1, y1, x2, y2)
            item.setPen(QPen(MEMO_COLOR, 0))
            item.setZValue(6)
            s.addItem(item)
        if self._memo_pending is not None:
            _marker(s, QPointF(*self._memo_pending), MEMO_COLOR, 3)
        if self._ref is not None:
            _marker(s, QPointF(*self._ref), REF_COLOR, 5, " 補正点")
        if self._sim is not None:  # クリックした点と距離のラベルは常に描く（カメラ設置座標が無限遠の垂直補正後でも）
            (x, y), text = self._sim
            if self._camera is not None:
                line = QGraphicsLineItem(self._camera[0], self._camera[1], x, y)
                line.setPen(QPen(SIM_COLOR, 0))
                line.setZValue(6)
                s.addItem(line)
            _marker(s, QPointF(x, y), SIM_COLOR, 5, f" {text}")
        self._update_scene_rect()

    def _update_scene_rect(self) -> None:
        rect = QRectF(self._image_rect)
        if self._camera is not None:
            rect = rect.united(QRectF(self._camera[0] - 50, self._camera[1] - 50, 100, 100))
        for _color, vp, segments, near in self._vp_draw_info():  # 消失点・補線の端も見えるようにする
            for point in ([vp] if near else []) + [seg[1] for seg in segments]:
                rect = rect.united(QRectF(point[0] - 50, point[1] - 50, 100, 100))
        margin = max(rect.width(), rect.height()) * 0.5
        self._scene.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

    def fit_all(self) -> None:
        """画像とカメラ設置座標が全部入るように表示を合わせる。"""
        if self._image_rect.isEmpty():
            return
        rect = QRectF(self._image_rect)
        if self._camera is not None:
            rect = rect.united(QRectF(self._camera[0] - 50, self._camera[1] - 50, 100, 100))
        pad = max(rect.width(), rect.height()) * 0.05
        self.fitInView(rect.adjusted(-pad, -pad, pad, pad), Qt.KeepAspectRatio)

    def set_points_visible(self, on: bool) -> None:
        """長方形の頂点（丸と番号）の表示・非表示。辺と塗りは残す。非表示の間は頂点のドラッグもできない。"""
        self._show_points = on
        self._rebuild()

    def set_antialiasing(self, on: bool) -> None:
        """線・点・文字の描画のアンチエイリアス（画像のスムージングとは別）。"""
        self.setRenderHint(QPainter.Antialiasing, on)
        self.setRenderHint(QPainter.TextAntialiasing, on)
        self.viewport().update()

    def fit_image(self) -> None:
        """画像だけを表示域の中心に、縦横どちらも収まる最大の倍率で表示する（カメラ設置座標は含めない）。"""
        if not self._image_rect.isEmpty():
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    # ---- 操作 ----
    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current = self.transform().m11()
        # 範囲外（カメラ設置座標が遠く、全体表示の倍率が下限未満など）からは、範囲内へ戻る向きの操作を許す
        if (factor > 1 and current < MAX_SCALE) or (factor < 1 and current > MIN_SCALE):
            self.scale(factor, factor)

    def _point_at(self, pos) -> int | None:
        """画面上の座標 pos の近くにある 4 点の番号（一番近いもの）。無ければ None。"""
        best, best_d = None, POINT_HIT_PX
        for i, (x, y) in enumerate(self._points if self._show_points else []):
            q = self.mapFromScene(QPointF(x, y))
            d = ((q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        return best

    def _vline_end_at(self, pos) -> tuple[int, int] | None:
        """画面上の座標 pos の近くにある垂直線の端点（線の番号, 0=始点/1=終点）。"""
        best, best_d = None, POINT_HIT_PX
        for i in range(len(self._vlines) - 1, -1, -1):
            for end, (x, y) in enumerate(self._vlines[i]):
                q = self.mapFromScene(QPointF(x, y))
                d = ((q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = (i, end), d
        return best

    def _memo_end_at(self, pos) -> tuple[int, int] | None:
        """画面上の座標 pos の近くにあるメモ線の端点（線の番号, 0=始点/1=終点）。後から引いた線を優先する。"""
        best, best_d = None, POINT_HIT_PX
        for i in range(len(self._memo_lines) - 1, -1, -1):
            for end, (x, y) in enumerate(self._memo_lines[i]):
                q = self.mapFromScene(QPointF(x, y))
                d = ((q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = (i, end), d
        return best

    def mousePressEvent(self, event) -> None:
        self._press_pos = event.position()
        self._dragged = False
        if event.button() == Qt.LeftButton and self.point_drag_enabled:
            self._drag_idx = self._point_at(event.position())
            if self._drag_idx is not None:
                event.accept()
                return
        if event.button() == Qt.LeftButton and self.vline_drag_enabled:
            self._drag_vline = self._vline_end_at(event.position())
            if self._drag_vline is not None:
                event.accept()
                return
        if event.button() == Qt.LeftButton and self.memo_drag_enabled:
            self._drag_memo = self._memo_end_at(event.position())
            if self._drag_memo is not None:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_vline is not None:
            p = self.mapToScene(event.position().toPoint())
            x = min(max(p.x(), self._image_rect.left()), self._image_rect.right())
            y = min(max(p.y(), self._image_rect.top()), self._image_rect.bottom())
            self.vline_moved.emit(*self._drag_vline, x, y)
            event.accept()
            return
        if self._drag_memo is not None:
            p = self.mapToScene(event.position().toPoint())
            x = min(max(p.x(), self._image_rect.left()), self._image_rect.right())
            y = min(max(p.y(), self._image_rect.top()), self._image_rect.bottom())
            self.memo_moved.emit(*self._drag_memo, x, y)
            event.accept()
            return
        if self._drag_idx is not None:
            p = self.mapToScene(event.position().toPoint())
            x = min(max(p.x(), self._image_rect.left()), self._image_rect.right())
            y = min(max(p.y(), self._image_rect.top()), self._image_rect.bottom())
            self.point_moved.emit(self._drag_idx, x, y)
            event.accept()
            return
        if event.buttons() == Qt.NoButton:
            hover = ((self.point_drag_enabled and self._point_at(event.position()) is not None)
                     or (self.memo_drag_enabled and self._memo_end_at(event.position()) is not None)
                     or (self.vline_drag_enabled and self._vline_end_at(event.position()) is not None))
            self.viewport().setCursor(Qt.SizeAllCursor if hover else Qt.ArrowCursor)
        if self._press_pos is not None and event.buttons() != Qt.NoButton:
            delta = event.position() - self._press_pos
            if self._dragged or abs(delta.x()) + abs(delta.y()) > DRAG_THRESHOLD_PX:
                self._dragged = True
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
                self._press_pos = event.position()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_vline is not None:
            self._drag_vline = None
            self._press_pos = None
            event.accept()
            return
        if self._drag_memo is not None:
            self._drag_memo = None
            self._press_pos = None
            event.accept()
            return
        if self._drag_idx is not None:
            self._drag_idx = None
            self._press_pos = None
            event.accept()
            return
        if event.button() == Qt.LeftButton and not self._dragged and not self._image_rect.isEmpty():
            p = self.mapToScene(event.position().toPoint())
            self.image_clicked.emit(p.x(), p.y())
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def image_contains(self, x: float, y: float) -> bool:
        return self._image_rect.contains(x, y)


class CameraProjectionScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_camera_projection_settings()
        self.points: list[tuple[float, float]] = []
        self.projection: CameraProjection | None = None
        self.image_size: tuple[int, int] | None = None
        self.ref_xy: tuple[float, float] | None = None   # 実測距離で補正する点
        self.orig_img = None                             # 元画像（BGR）
        self.dist: Distortion | None = None              # レンズ歪みの補正（4 点などは補正後の画像の座標で持つ）
        self.dist_lines: list[list[tuple[float, float]]] = []  # 歪み補正線（元画像の座標）
        self.memo_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []  # メモ線（元画像の座標・保存しない）
        self.memo_pending: tuple[float, float] | None = None
        self.focal_override: float | None = None         # 補正後の焦点距離（None = 自動推定）
        self.rect: VerticalRectification | None = None  # 垂直補正（歪み補正の後。4 点などは垂直補正後の画像の座標で持つ）
        self.vlines: list[tuple[tuple[float, float], tuple[float, float]]] = []  # 建物の垂直線（元画像の座標）
        self.vpending: tuple[float, float] | None = None

        root = QVBoxLayout(self)
        title = QLabel("射影変換設定抽出")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)
        root.addWidget(QLabel(
            "画像上で、現実では長方形になる 4 点を長方形を一周する順にクリックし、2 辺の長さを入力して「カメラ位置算出」を押す。"
            "（左ドラッグ: 移動（4 点の上ではその点を動かす） / ホイール: 拡大縮小。カメラ設置座標は画像の外になることがある）"))

        self.image_edit = QLineEdit(self.settings["image_path"])
        self.image_edit.setReadOnly(True)
        add_path_row(root, "カメラ画像ファイル:", self.image_edit, self._pick_image)
        self.out_edit = QLineEdit(self.settings["output_dir"])
        self.out_edit.setReadOnly(True)
        add_path_row(root, "設定ファイル出力先:", self.out_edit, self._pick_output)

        self.view = ImageView()
        root.addWidget(self.view, 1)

        row = QHBoxLayout()
        row.addWidget(fixed_label("長方形の長辺:"))
        self.long_spin = self._side_spin()
        row.addWidget(self.long_spin)
        row.addWidget(fixed_label("短辺:", 50))
        self.short_spin = self._side_spin()
        row.addWidget(self.short_spin)
        row.addWidget(QLabel("1→2 の辺は"))
        self.first_combo = QComboBox()
        self.first_combo.addItem("長辺", "long")
        self.first_combo.addItem("短辺", "short")
        row.addWidget(self.first_combo)
        self.btn_undo = QPushButton("最後の点を戻す")
        self.btn_clear = QPushButton("点をクリア")
        self.btn_fit = QPushButton("全体表示")
        self.btn_fit_image = QPushButton("画像中心表示")
        self.chk_horizon = QCheckBox("消失点・地平線を表示")
        self.chk_antialias = QCheckBox("アンチエイリアス")
        self.chk_points = QCheckBox("長方形の頂点を表示")
        for b in (self.btn_undo, self.btn_clear, self.btn_fit, self.btn_fit_image):
            row.addWidget(b)
        row.addWidget(self.chk_points)
        row.addWidget(self.chk_horizon)
        row.addWidget(self.chk_antialias)
        row.addStretch()
        root.addLayout(row)

        mrow = QHBoxLayout()
        self.btn_memo = QPushButton("メモ線を描画")
        self.btn_memo.setCheckable(True)
        self.btn_undo_memo = QPushButton("メモ線を戻す")
        self.btn_clear_memo = QPushButton("メモ線をクリア")
        mrow.addWidget(QLabel("メモ（一時的・保存しません）:"))
        for w in (self.btn_memo, self.btn_undo_memo, self.btn_clear_memo):
            mrow.addWidget(w)
        mrow.addStretch()
        root.addLayout(mrow)

        vrow = QHBoxLayout()
        self.btn_vline = QPushButton("建物の垂直線を指定")
        self.btn_vline.setCheckable(True)
        self.btn_undo_vline = QPushButton("垂直線を戻す")
        self.btn_clear_vlines = QPushButton("垂直線をクリア")
        self.btn_apply_rect = QPushButton("垂直補正を適用")
        self.btn_reset_rect = QPushButton("垂直補正解除")
        vrow.addWidget(QLabel("垂直補正:"))
        for w in (self.btn_vline, self.btn_undo_vline, self.btn_clear_vlines, self.btn_apply_rect, self.btn_reset_rect):
            vrow.addWidget(w)
        vrow.addStretch()
        root.addLayout(vrow)
        self.rect_label = QLabel("")
        root.addWidget(self.rect_label)

        drow = QHBoxLayout()
        self.btn_dline = QPushButton("歪み補正線を指定")
        self.btn_dline.setCheckable(True)
        self.btn_newline = QPushButton("新しい線")
        self.btn_undo_dline = QPushButton("線の最後の点を戻す")
        self.btn_clear_dlines = QPushButton("線をクリア")
        self.btn_estimate = QPushButton("歪み係数を推定")
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setDecimals(3)
        self.lambda_spin.setSingleStep(0.01)
        self.lambda_spin.setPrefix("λ = ")
        self.lambda_spin.setKeyboardTracking(False)
        self.lambda_spin.setRange(-0.8, 0.2)
        self.btn_reset_dist = QPushButton("歪み補正解除")
        drow.addWidget(QLabel("歪み補正:"))
        for w in (self.btn_dline, self.btn_newline, self.btn_undo_dline, self.btn_clear_dlines, self.btn_estimate,
                  self.lambda_spin, self.btn_reset_dist):
            drow.addWidget(w)
        drow.addStretch()
        root.addLayout(drow)
        self.dist_label = QLabel("")
        root.addWidget(self.dist_label)

        crow = QHBoxLayout()
        self.btn_ref = QPushButton("補正点を指定")
        self.btn_ref.setCheckable(True)
        self.ref_label = QLabel("")
        self.measured_spin = self._side_spin()
        self.btn_calibrate = QPushButton("実測距離で補正")
        self.btn_reset_cal = QPushButton("補正解除")
        crow.addWidget(QLabel("実測補正:"))
        crow.addWidget(self.btn_ref)
        crow.addWidget(self.ref_label)
        crow.addWidget(QLabel("カメラ直下からの実測距離（地面上）"))
        crow.addWidget(self.measured_spin)
        crow.addWidget(self.btn_calibrate)
        crow.addWidget(self.btn_reset_cal)
        crow.addStretch()
        root.addLayout(crow)

        self.points_label = QLabel("")
        root.addWidget(self.points_label)

        brow = QHBoxLayout()
        self.btn_back = QPushButton("戻る")
        self.btn_calc = QPushButton("カメラ位置算出")
        self.btn_sim = QPushButton("距離シミュレーション")
        self.btn_sim.setCheckable(True)
        self.btn_save = QPushButton("設定保存")
        brow.addWidget(self.btn_back)
        for b in (self.btn_calc, self.btn_sim, self.btn_save):
            brow.addWidget(b, 1)
        root.addLayout(brow)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.view.image_clicked.connect(self._on_image_clicked)
        self.view.point_moved.connect(self._on_point_moved)
        self.view.memo_moved.connect(self._on_memo_moved)
        self.view.vline_moved.connect(self._on_vline_moved)
        self.long_spin.valueChanged.connect(self._invalidate)
        self.short_spin.valueChanged.connect(self._invalidate)
        self.first_combo.currentIndexChanged.connect(self._invalidate)
        self.btn_undo.clicked.connect(self._undo_point)
        self.btn_clear.clicked.connect(self._clear_points)
        self.btn_fit.clicked.connect(self.view.fit_all)
        self.btn_fit_image.clicked.connect(self.view.fit_image)
        self.chk_horizon.setChecked(self.settings["show_guides"])  # 前回の表示設定を復元（接続前なので保存は走らない）
        self.chk_antialias.setChecked(self.settings["antialias"])
        self.view.set_antialiasing(self.chk_antialias.isChecked())
        self.chk_points.setChecked(self.settings["show_points"])
        self.view.set_points_visible(self.chk_points.isChecked())
        self.chk_points.toggled.connect(self.view.set_points_visible)
        self.chk_points.toggled.connect(self._save_screen_settings)
        self.chk_horizon.toggled.connect(self._update_guides)
        self.chk_antialias.toggled.connect(self.view.set_antialiasing)
        self.chk_horizon.toggled.connect(self._save_screen_settings)
        self.chk_antialias.toggled.connect(self._save_screen_settings)
        self.btn_calc.clicked.connect(self._on_calc)
        self.btn_sim.toggled.connect(self._on_sim_toggled)
        self.btn_ref.toggled.connect(self._on_ref_toggled)
        self.btn_dline.toggled.connect(self._on_dline_toggled)
        self.btn_memo.toggled.connect(self._on_memo_toggled)
        self.btn_vline.toggled.connect(self._on_vline_toggled)
        self.btn_undo_vline.clicked.connect(self._on_undo_vline)
        self.btn_clear_vlines.clicked.connect(self._on_clear_vlines)
        self.btn_apply_rect.clicked.connect(self._on_apply_rect)
        self.btn_reset_rect.clicked.connect(self._on_reset_rect)
        self.btn_undo_memo.clicked.connect(self._on_undo_memo)
        self.btn_clear_memo.clicked.connect(self._on_clear_memo)
        self.btn_newline.clicked.connect(self._on_new_dist_line)
        self.btn_undo_dline.clicked.connect(self._on_undo_dist_point)
        self.btn_clear_dlines.clicked.connect(self._on_clear_dist_lines)
        self.btn_estimate.clicked.connect(self._on_estimate_distortion)
        self.lambda_spin.valueChanged.connect(self._on_lambda_changed)
        self.btn_reset_dist.clicked.connect(lambda: self._set_lambda(0.0))
        self.measured_spin.valueChanged.connect(self._refresh_state)
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        self.btn_reset_cal.clicked.connect(self._on_reset_calibration)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_back.clicked.connect(self.back_requested.emit)

        self._refresh_state()
        if self.settings["image_path"] and Path(self.settings["image_path"]).is_file():
            self._load_image(Path(self.settings["image_path"]))

    @staticmethod
    def _side_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, MAX_SIDE_M)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setSuffix(" m")
        return spin

    # ---- パス・画像 ----
    def _pick_image(self) -> None:
        current = self.image_edit.text()
        start = str(Path(current).parent) if current else (
            str(CAMERA_PROJECTION_IMAGE_DIR) if CAMERA_PROJECTION_IMAGE_DIR.is_dir() else "")
        path, _ = QFileDialog.getOpenFileName(self, "カメラ画像ファイルを選択", start, IMAGE_FILTER)
        if path:
            self._load_image(Path(path))

    def _pick_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "設定ファイル出力先フォルダを選択", self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self._save_screen_settings()
            if self.image_size is not None:
                self._load_saved_config()

    def _save_screen_settings(self, *_args) -> None:
        self.settings.update(image_path=self.image_edit.text(), output_dir=self.out_edit.text(),
                             show_guides=self.chk_horizon.isChecked(), antialias=self.chk_antialias.isChecked(),
                             show_points=self.chk_points.isChecked())
        save_camera_projection_settings(self.settings)

    def _load_image(self, path: Path) -> None:
        img = imread(path)
        if img is None:
            QMessageBox.warning(self, "射影変換設定抽出", f"画像を読み込めません:\n{path}")
            return
        h, w = img.shape[:2]
        qimage = self._to_qimage(img)
        self.image_edit.setText(str(path))
        self.image_size = (w, h)
        self.orig_img = img
        self.dist = Distortion(0.0, (w, h))
        self.dist_lines = []
        self._set_lambda_spin(0.0)
        self.btn_dline.setChecked(False)
        self.points = []
        self.projection = None
        self.ref_xy = None
        self.focal_override = None
        self.btn_sim.setChecked(False)
        self.btn_ref.setChecked(False)
        self.btn_memo.setChecked(False)
        self.btn_vline.setChecked(False)
        self.memo_lines, self.memo_pending = [], None
        self.rect, self.vlines, self.vpending = None, [], None
        self.view.set_image(qimage)
        self._save_screen_settings()
        self._load_saved_config()
        self._refresh_state()

    @staticmethod
    def _to_qimage(img) -> QImage:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _config_path(self) -> Path | None:
        if not self.image_edit.text() or not self.out_edit.text():
            return None
        return Path(self.out_edit.text()) / (Path(self.image_edit.text()).stem + ".json")

    def _load_saved_config(self) -> None:
        """出力先に前回保存した <画像名>.json があれば、4 点・辺の長さ・カメラ設置座標を復元する。"""
        path = self._config_path()
        cfg = load_config(path) if path is not None and path.is_file() else None
        if cfg is None:
            self.status.setText("")
            return
        dist = cfg["distortion"]  # 4 点などは補正後の画像の座標で保存されている（変換はせず、補正後の画像に切り替える）
        self.dist = Distortion(dist["lambda"] if dist else 0.0, self.image_size)
        self.dist_lines = [list(line) for line in dist["lines"]] if dist else []
        self._set_lambda_spin(self.dist.lam)
        self.rect, self.vlines, self.vpending = None, [], None
        rc = cfg["rectification"]  # 垂直補正（垂直線と焦点距離から作り直す。点は垂直補正後の座標で保存されている）
        if rc:
            self.vlines = [tuple(line) for line in rc["lines"]]
            try:
                self.rect = self._build_rect(self.vlines, rc["focal_px"])
            except ProjectionError as e:
                self.status.setText(f"垂直補正を復元できませんでした: {e}")
        self._show_working_image()
        self._refresh_dist_lines()
        self._refresh_memo()
        self._refresh_vlines()
        self.points = list(cfg["points"])
        self.long_spin.blockSignals(True)
        self.short_spin.blockSignals(True)
        self.first_combo.blockSignals(True)
        self.long_spin.setValue(cfg["long_side_m"])
        self.short_spin.setValue(cfg["short_side_m"])
        self.first_combo.setCurrentIndex(self.first_combo.findData(cfg["first_side"]))
        self.long_spin.blockSignals(False)
        self.short_spin.blockSignals(False)
        self.first_combo.blockSignals(False)
        self.view.set_points(self.points)
        cal = cfg["calibration"]
        self.focal_override = cal["focal_px"] if cal else None
        self.ref_xy = cal["ref_xy"] if cal else None
        self.view.set_ref(self.ref_xy)
        if cal:
            self.measured_spin.blockSignals(True)
            self.measured_spin.setValue(cal["measured_m"])
            self.measured_spin.blockSignals(False)
        self._compute(notify=False)
        self.status.setText(f"前回の設定を読み込みました: {path.name}\n{self.status.text()}")
        self._refresh_state()

    # ---- 点の指定 ----
    def _on_image_clicked(self, x: float, y: float) -> None:
        if self.btn_vline.isChecked():
            if self.view.image_contains(x, y) and self.dist is not None:
                o = self._from_view((x, y))
                if self.vpending is None:
                    self.vpending = o
                else:
                    self.vlines.append((self.vpending, o))
                    self.vpending = None
                self._refresh_vlines()
                self._refresh_state()
            return
        if self.btn_memo.isChecked():
            if self.view.image_contains(x, y) and self.dist is not None:
                o = self._from_view((x, y))
                if self.memo_pending is None:
                    self.memo_pending = o
                else:
                    self.memo_lines.append((self.memo_pending, o))
                    self.memo_pending = None
                self._refresh_memo()
            return
        if self.btn_dline.isChecked():
            if self.view.image_contains(x, y) and self.dist is not None:
                if not self.dist_lines:
                    self.dist_lines.append([])
                self.dist_lines[-1].append(self._from_view((x, y)))
                self._refresh_dist_lines()
                self._refresh_state()
            return
        if self.btn_ref.isChecked():
            if self.view.image_contains(x, y):
                self.ref_xy = (x, y)
                self.view.set_ref(self.ref_xy)
                self.btn_ref.setChecked(False)
                self._refresh_state()
            return
        if self.btn_sim.isChecked() and self.projection is not None:
            dist = self.projection.distance_m((x, y))
            slant = self.projection.slant_distance_m((x, y))
            self.view.set_sim(((x, y), f"地面 {dist:.2f} m / 直線 {slant:.2f} m"))
            self.status.setText(
                f"クリック位置 ({x:.0f}, {y:.0f}) px{self._corrected_note()}: カメラ設置地点（直下）から 地面上 {dist:.2f} m / "
                f"カメラ本体からの直線距離 {slant:.2f} m（推定カメラ高さ {self.projection.height_m:.2f} m）")
            return
        if len(self.points) >= 4 or not self.view.image_contains(x, y):
            return
        self.points.append((x, y))
        self._invalidate()
        self.view.set_points(self.points)

    def _on_point_moved(self, idx: int, x: float, y: float) -> None:
        """ドラッグした 4 点の位置を更新する。算出済みの結果は破棄する（再度「カメラ位置算出」）。"""
        if idx >= len(self.points):
            return
        self.points[idx] = (x, y)
        self._invalidate()
        self.view.set_points(self.points)

    def _undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self._invalidate()
            self.view.set_points(self.points)

    def _clear_points(self) -> None:
        self.points = []
        self._invalidate()
        self.view.set_points(self.points)

    # ---- 算出・シミュレーション・保存 ----
    def _sides(self) -> tuple[float, float]:
        return resolve_sides(self.long_spin.value(), self.short_spin.value(), self.first_combo.currentData())

    def _inputs_ready(self) -> bool:
        return (self.image_size is not None and len(self.points) == 4
                and self.short_spin.value() > 0 and self.long_spin.value() >= self.short_spin.value())

    def _invalidate(self, *_args) -> None:
        """入力が変わったら算出結果を捨てる。"""
        self.projection = None
        self.focal_override = None
        self.btn_sim.setChecked(False)
        self.view.set_camera(None)
        self.view.set_sim(None)
        self._update_guides()
        self._refresh_state()

    def _refresh_state(self) -> None:
        self.points_label.setText(
            f"指定した点: {len(self.points)}/4  " + "  ".join(f"{i + 1}:({x:.0f}, {y:.0f})" for i, (x, y) in enumerate(self.points)))
        self.btn_calc.setEnabled(self._inputs_ready())
        self.btn_sim.setEnabled(self.projection is not None)
        self.btn_save.setEnabled(self.projection is not None)
        self.btn_ref.setEnabled(self.image_size is not None)
        self.ref_label.setText(f"({self.ref_xy[0]:.0f}, {self.ref_xy[1]:.0f})" if self.ref_xy else "未指定")
        self.btn_calibrate.setEnabled(self._inputs_ready() and self.ref_xy is not None and self.measured_spin.value() > 0)
        self.btn_reset_cal.setEnabled(self.focal_override is not None)
        rect_on = self.rect is not None
        self._update_cursor()  # 垂直補正の適用有無で垂直線の端点のドラッグ可否が変わる
        if rect_on:
            self.btn_calibrate.setEnabled(False)  # 垂直補正後の距離は焦点距離に依らない（H だけで決まる）
        has_v = bool(self.vlines) or self.vpending is not None
        self.btn_vline.setEnabled(self.dist is not None and not rect_on)
        self.btn_undo_vline.setEnabled(has_v and not rect_on)
        self.btn_clear_vlines.setEnabled(has_v and not rect_on)
        self.btn_apply_rect.setEnabled(self.dist is not None and not rect_on and len(self.vlines) >= MIN_VERTICAL_LINES
                                       and self.projection is not None)
        self.btn_reset_rect.setEnabled(rect_on)
        if rect_on:
            state = (f"垂直補正中（縮尺 {self.rect.scale:.2f}。以降の座標は垂直補正後の画像の座標。"
                     "カメラ直下は画像の下方向の無限遠で、距離は地面座標から計算）")
        else:
            state = f"補正なし（適用には垂直線 {MIN_VERTICAL_LINES} 本以上と、焦点距離を求めるための「カメラ位置算出」が必要）"
        self.rect_label.setText(f"建物の垂直線: {len(self.vlines)} 本 / {state}")
        self.btn_undo.setEnabled(bool(self.points))
        self.btn_clear.setEnabled(bool(self.points))
        loaded = self.dist is not None
        self.btn_dline.setEnabled(loaded)
        self.btn_newline.setEnabled(loaded and bool(self.dist_lines) and bool(self.dist_lines[-1]))
        self.btn_undo_dline.setEnabled(any(self.dist_lines))
        self.btn_clear_dlines.setEnabled(bool(self.dist_lines))
        self.btn_estimate.setEnabled(any(len(line) >= MIN_LINE_POINTS for line in self.dist_lines))
        self.lambda_spin.setEnabled(loaded)
        self.btn_reset_dist.setEnabled(loaded and not self.dist.is_identity)
        n_pts = ", ".join(str(len(line)) for line in self.dist_lines)
        applied = (f"補正中: λ = {self.dist.lam:.3f}（縮尺 {self.dist.scale:.2f}。以降の座標は補正後の画像の座標）"
                   if loaded and not self.dist.is_identity else "補正なし")
        self.dist_label.setText(f"歪み補正線: {len(self.dist_lines)} 本（点数: {n_pts or '-'}） / {applied}")

    def _on_calc(self) -> None:
        self._compute(notify=True)
        self._refresh_state()

    def _compute(self, notify: bool) -> None:
        if not self._inputs_ready():
            return
        side_a, side_b = self._sides()
        try:
            focal = self.focal_override or (self.rect.focal_px if self.rect is not None else None)
            proj = compute_projection(self.points, side_a, side_b, self.image_size, focal_px=focal,
                                      vertical=self.rect.vertical if self.rect is not None else None)
        except ProjectionError as e:
            self.projection = None
            self.view.set_camera(None)
            self.status.setText(f"算出できませんでした: {e}")
            if notify:
                QMessageBox.warning(self, "射影変換設定抽出", str(e))
            return
        self.projection = proj
        self._update_guides()
        self.view.set_camera(proj.camera_xy)
        self.view.fit_all()
        if proj.camera_xy is None:
            gx, gy = proj.camera_ground
            head = f"カメラ設置座標: 画像の下方向の無限遠（垂直補正済み。カメラ直下の地面座標 ({gx:.2f}, {gy:.2f}) m）"
        else:
            cx, cy = proj.camera_xy
            w, h = self.image_size
            where = "（画像の外）" if not (0 <= cx < w and 0 <= cy < h) else ""
            head = f"カメラ設置座標: ({cx:.0f}, {cy:.0f}) px{where}"
        kind = "垂直補正後の" if self.rect is not None else ("補正後" if self.focal_override else "推定")
        self.status.setText(
            f"{head} / 推定カメラ高さ: {proj.height_m:.2f} m / {kind}焦点距離: {proj.focal_px:.0f} px"
            "\n「距離シミュレーション」を押して画像をクリックすると、カメラ設置地点からの距離を表示します。")

    def _update_guides(self, *_args) -> None:
        """チェックが入っていて算出済みなら、消失点 2 つ・その補線（長方形の辺の延長）・地平線（2 つの消失点を結ぶ直線）を描く。"""
        horizon, vanishing = None, None
        if self.chk_horizon.isChecked() and self.projection is not None and self.image_size is not None:
            vanishing = self.projection.vanishing_points
            horizon = clip_line_to_rect(vanishing[0], vanishing[1], *self.image_size)
        self.view.set_guides(horizon, vanishing)

    def _on_sim_toggled(self, on: bool) -> None:
        if on:
            self.btn_vline.setChecked(False)
            self.btn_memo.setChecked(False)
            self.btn_ref.setChecked(False)
            self.btn_dline.setChecked(False)
        self._update_cursor()
        if not on:
            self.view.set_sim(None)

    def _on_dline_toggled(self, on: bool) -> None:
        if on:
            self.btn_vline.setChecked(False)
            self.btn_memo.setChecked(False)
            self.btn_sim.setChecked(False)
            self.btn_ref.setChecked(False)
        self._update_cursor()

    def _on_ref_toggled(self, on: bool) -> None:
        if on:
            self.btn_vline.setChecked(False)
            self.btn_memo.setChecked(False)
            self.btn_sim.setChecked(False)
            self.btn_dline.setChecked(False)
        self._update_cursor()

    def _update_cursor(self) -> None:
        placing = (self.btn_sim.isChecked() or self.btn_ref.isChecked() or self.btn_dline.isChecked()
                   or self.btn_vline.isChecked())
        crosshair = placing or self.btn_memo.isChecked()
        self.view.point_drag_enabled = not crosshair
        self.view.memo_drag_enabled = not placing
        self.view.vline_drag_enabled = not (placing and not self.btn_vline.isChecked()) and self.rect is None
        self.view.setCursor(Qt.CrossCursor if crosshair else Qt.ArrowCursor)

    def _on_calibrate(self) -> None:
        """補正点までの実測距離（カメラ直下から）に合うよう焦点距離を調整し、カメラ設置座標を求め直す。"""
        if self.rect is not None:
            QMessageBox.information(self, "射影変換設定抽出", "垂直補正後は距離が焦点距離に依らないため、実測距離による補正は使えません。"
                                    "使うときは「垂直補正解除」を押してください。")
            return
        side_a, side_b = self._sides()
        try:
            proj = calibrate_focal(self.points, side_a, side_b, self.image_size, self.ref_xy, self.measured_spin.value())
        except ProjectionError as e:
            self.status.setText(f"補正できませんでした: {e}")
            QMessageBox.warning(self, "射影変換設定抽出", str(e))
            return
        self.focal_override = proj.focal_px
        self.btn_sim.setChecked(False)
        self.view.set_sim(None)
        self._compute(notify=False)
        self.status.setText("実測距離で補正しました。\n" + self.status.text())
        self._refresh_state()

    def _on_reset_calibration(self) -> None:
        self.focal_override = None
        self.btn_sim.setChecked(False)
        self.view.set_sim(None)
        self._compute(notify=False)
        self._refresh_state()

    # ---- メモ線（一時的。保存しない。座標は元画像で持ち、歪み係数を変えても同じ位置に付く） ----
    def _refresh_memo(self) -> None:
        if self.dist is None:
            return
        conv = self._to_view
        self.view.set_memo([(conv(a), conv(b)) for a, b in self.memo_lines],
                           conv(self.memo_pending) if self.memo_pending is not None else None)

    def _on_memo_toggled(self, on: bool) -> None:
        if on:
            self.btn_vline.setChecked(False)
            self.btn_sim.setChecked(False)
            self.btn_ref.setChecked(False)
            self.btn_dline.setChecked(False)
        else:
            self.memo_pending = None  # 始点だけの状態は破棄
            self._refresh_memo()
        self._update_cursor()

    def _on_memo_moved(self, idx: int, end: int, x: float, y: float) -> None:
        """ドラッグしたメモ線の端点を更新する（元画像の座標で持つ）。"""
        if self.dist is None or idx >= len(self.memo_lines):
            return
        line = list(self.memo_lines[idx])
        line[end] = self._from_view((x, y))
        self.memo_lines[idx] = (line[0], line[1])
        self._refresh_memo()

    def _on_undo_memo(self) -> None:
        if self.memo_pending is not None:
            self.memo_pending = None
        elif self.memo_lines:
            self.memo_lines.pop()
        self._refresh_memo()

    def _on_clear_memo(self) -> None:
        self.memo_lines, self.memo_pending = [], None
        self._refresh_memo()

    # ---- レンズ歪みの補正 ----
    def _corrected_note(self) -> str:
        on = self.rect is not None or (self.dist is not None and not self.dist.is_identity)
        return "（補正後の画像の座標）" if on else ""

    # ---- 座標系: 元画像 →（歪み補正）→（垂直補正）→ 表示中の画像。元画像の座標で持つ線は変換して表示する ----
    def _to_view(self, p) -> tuple[float, float]:
        q = self.dist.to_corrected(p)
        if self.rect is not None:
            q = self.rect.to_rectified(q)
        return float(q[0]), float(q[1])

    def _from_view(self, p) -> tuple[float, float]:
        q = p
        if self.rect is not None:
            q = self.rect.to_unrectified(q)
        o = self.dist.to_original(q)
        return float(o[0]), float(o[1])

    def _set_lambda_spin(self, lam: float) -> None:
        """λ の入力欄を画像サイズで使える範囲にして値を入れる（変更通知は出さない）。"""
        lo, hi = lambda_range(self.image_size)
        self.lambda_spin.blockSignals(True)
        self.lambda_spin.setRange(math.ceil(lo * 1000) / 1000, math.floor(hi * 1000) / 1000)
        self.lambda_spin.setValue(lam)
        self.lambda_spin.blockSignals(False)

    def _show_working_image(self) -> None:
        """補正後の画像（補正なしなら元画像）を表示する。"""
        if self.orig_img is not None and self.dist is not None:
            img = self.dist.correct_image(self.orig_img)
            if self.rect is not None:
                img = self.rect.rectify_image(img)
            self.view.set_background_image(self._to_qimage(img))

    def _refresh_dist_lines(self) -> None:
        if self.dist is None:
            return
        self.view.set_dist_lines([[self._to_view(p) for p in line] for line in self.dist_lines])

    def _on_lambda_changed(self, value: float) -> None:
        self._set_lambda(value)

    def _set_lambda(self, lam: float) -> None:
        """歪み係数を変えて補正後の画像に切り替える。指定済みの 4 点・補正点は元画像の位置を保ったまま付け替え、算出結果は破棄する。"""
        if self.orig_img is None or self.dist is None:
            return
        orig_points = [self._from_view(p) for p in self.points]
        orig_ref = self._from_view(self.ref_xy) if self.ref_xy is not None else None
        had_rect = self.rect is not None
        self.rect = None  # 垂直補正は歪み補正後の画像から求めたものなので、歪み係数を変えたら解除する（垂直線は残す）
        new = Distortion(lam, self.image_size)
        self.dist = new
        self.points = [self._to_view(p) for p in orig_points]
        if orig_ref is not None:
            self.ref_xy = self._to_view(orig_ref)
        self._set_lambda_spin(new.lam)
        self._show_working_image()
        self._invalidate()
        self.view.set_points(self.points)
        self.view.set_ref(self.ref_xy)
        self._refresh_dist_lines()
        self._refresh_memo()
        self._refresh_vlines()
        self._refresh_state()
        if had_rect:
            self.status.setText("歪み係数を変えたので垂直補正を解除しました。必要なら「垂直補正を適用」をやり直してください。")

    # ---- 垂直補正（建物の垂直線） ----
    def _refresh_vlines(self) -> None:
        if self.dist is None:
            return
        self.view.set_vlines([(self._to_view(a), self._to_view(b)) for a, b in self.vlines],
                             self._to_view(self.vpending) if self.vpending is not None else None)

    def _on_vline_toggled(self, on: bool) -> None:
        if on:
            for b in (self.btn_sim, self.btn_ref, self.btn_dline, self.btn_memo):
                b.setChecked(False)
        else:
            self.vpending = None  # 始点だけの状態は破棄
            self._refresh_vlines()
        self._update_cursor()

    def _on_vline_moved(self, idx: int, end: int, x: float, y: float) -> None:
        """ドラッグした垂直線の端点を更新する（元画像の座標で持つ）。適用前に限る。"""
        if self.dist is None or self.rect is not None or idx >= len(self.vlines):
            return
        line = list(self.vlines[idx])
        line[end] = self._from_view((x, y))
        self.vlines[idx] = (line[0], line[1])
        self._refresh_vlines()

    def _on_undo_vline(self) -> None:
        if self.vpending is not None:
            self.vpending = None
        elif self.vlines:
            self.vlines.pop()
        self._refresh_vlines()
        self._refresh_state()

    def _on_clear_vlines(self) -> None:
        self.vlines, self.vpending = [], None
        self._refresh_vlines()
        self._refresh_state()

    def _build_rect(self, lines, focal_px: float) -> VerticalRectification:
        """垂直線（元画像の座標）を歪み補正後の座標に直して消失点を求め、垂直補正を作る。"""
        lens = [(self.dist.to_corrected(a), self.dist.to_corrected(b)) for a, b in lines]
        return VerticalRectification(vertical_vanishing_point(lens, self.image_size), focal_px, self.image_size)

    def _set_rect(self, rect: VerticalRectification | None) -> None:
        """垂直補正を切り替える。4 点・補正点は元画像の位置を保ったまま付け替え、算出結果は破棄する。"""
        orig_points = [self._from_view(p) for p in self.points]
        orig_ref = self._from_view(self.ref_xy) if self.ref_xy is not None else None
        self.rect = rect
        self.points = [self._to_view(p) for p in orig_points]
        if orig_ref is not None:
            self.ref_xy = self._to_view(orig_ref)
        self.btn_vline.setChecked(False)
        self._update_cursor()
        self._show_working_image()
        self._invalidate()  # 算出結果・焦点距離の補正・シミュレーションを捨てる
        self.view.set_points(self.points)
        self.view.set_ref(self.ref_xy)
        self._refresh_dist_lines()
        self._refresh_memo()
        self._refresh_vlines()
        self._refresh_state()

    def _on_apply_rect(self) -> None:
        """垂直線の消失点から、垂直線が縦方向に平行になる画像へ切り替える。焦点距離は算出済みの値を使う。"""
        if self.projection is None:
            return
        try:
            rect = self._build_rect(self.vlines, self.projection.focal_px)
        except ProjectionError as e:
            self.status.setText(f"垂直補正できませんでした: {e}")
            QMessageBox.warning(self, "射影変換設定抽出", str(e))
            return
        self._set_rect(rect)
        tilts = [line_tilt_deg(self._to_view(a), self._to_view(b)) for a, b in self.vlines]
        note = f"垂直補正を適用しました（縮尺 {rect.scale:.2f}。垂直線の縦方向からの傾き 最大 {max(tilts):.2f}°）。"
        if rect.cropped:
            note += "画像の端は補正後の画像からはみ出して切れています（座標の計算には影響しません）。"
        if self._inputs_ready():
            self._compute(notify=False)
            self.status.setText(f"{note}\n{self.status.text()}")
        else:
            self.status.setText(note + "\n4 点を指定し直して「カメラ位置算出」を押してください。")
        self._refresh_state()

    def _on_reset_rect(self) -> None:
        self._set_rect(None)
        if self._inputs_ready():
            self._compute(notify=False)  # 算出済みに戻す（そのまま「垂直補正を適用」を押し直せる）
            self.status.setText(f"垂直補正を解除しました。\n{self.status.text()}")
        else:
            self.status.setText("垂直補正を解除しました。4 点を指定して「カメラ位置算出」を押してください。")
        self._refresh_state()

    def _on_new_dist_line(self) -> None:
        if not self.dist_lines or self.dist_lines[-1]:
            self.dist_lines.append([])
        self._refresh_state()

    def _on_undo_dist_point(self) -> None:
        while self.dist_lines and not self.dist_lines[-1]:
            self.dist_lines.pop()
        if self.dist_lines:
            self.dist_lines[-1].pop()
        self._refresh_dist_lines()
        self._refresh_state()

    def _on_clear_dist_lines(self) -> None:
        self.dist_lines = []
        self._refresh_dist_lines()
        self._refresh_state()

    def _on_estimate_distortion(self) -> None:
        """現実ではまっすぐな線（歪み補正線）が補正後にまっすぐになる歪み係数を推定して、補正後の画像に切り替える。"""
        try:
            lam = estimate_lambda(self.dist_lines, self.image_size)
        except ProjectionError as e:
            self.status.setText(f"歪み係数を推定できませんでした: {e}")
            QMessageBox.warning(self, "射影変換設定抽出", str(e))
            return
        self._set_lambda(lam)
        self.status.setText(f"歪み係数を推定しました: λ = {lam:.3f}。線がまっすぐになっているか確認し、"
                            "4 点を指定（指定済みの 4 点は補正後の位置に移動）して「カメラ位置算出」を押してください。")

    def _on_save(self) -> None:
        path = self._config_path()
        if path is None or self.projection is None:
            QMessageBox.warning(self, "射影変換設定抽出", "設定ファイル出力先フォルダを指定してください。")
            return
        try:
            extra = {}
            if self.focal_override:
                extra.update(focal_px=self.focal_override, ref_xy=self.ref_xy, measured_m=self.measured_spin.value())
            if self.dist is not None and not self.dist.is_identity:
                extra["distortion"] = {"lambda": self.dist.lam, "scale": self.dist.scale,
                                       "lines": [line for line in self.dist_lines if line]}
            if self.rect is not None:
                extra["rectification"] = {"lines": [line for line in self.vlines], "focal_px": self.rect.source_focal_px}
            save_config(path, Path(self.image_edit.text()).name, self.points, self.long_spin.value(),
                        self.short_spin.value(), self.first_combo.currentData(), self.projection.camera_xy, **extra)
        except OSError as e:
            QMessageBox.critical(self, "射影変換設定抽出", f"保存に失敗しました:\n{e}")
            return
        self._save_screen_settings()
        QMessageBox.information(self, "射影変換設定抽出", f"設定を保存しました。\n{path}")
