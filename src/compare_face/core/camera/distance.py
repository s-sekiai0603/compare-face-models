"""カメラ設定ファイル（<カメラID>.json。射影変換設定抽出画面の出力）を使った、足元座標 → カメラからの距離(m)の算出（GUI 非依存）。

設定ファイルの 4 点などは歪み補正・垂直補正後の画像の座標で保存されているので、足元の座標（元のフレーム座標）は
元画像 →（歪み補正）→（垂直補正）の順に変換してから projective.CameraProjection.distance_m に渡す
（ui/camera_projection_screen.py の _to_view と同じ手順）。
設定ファイルには画像サイズが無いので、カメラ画像フォルダのカメラ画像（ファイル名 = カメラID.png、大文字小文字無視）から読む
（カメラ画像フォルダを指定しなければ設定フォルダ内から探す）。画像が無い・設定が読めない・算出できないカメラは距離なし（None）。"""
from __future__ import annotations

from pathlib import Path

from ..common.imageio import imread
from .distortion import Distortion
from .projective import CameraProjection, ProjectionError, compute_projection, load_config, resolve_sides
from .vertical import VerticalRectification, vertical_vanishing_point

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class _Camera:
    """1 台のカメラの座標変換と射影変換。"""

    def __init__(self, cfg: dict, size: tuple[int, int]):
        dist = cfg["distortion"]
        self.dist = Distortion(dist["lambda"] if dist else 0.0, size)
        self.rect: VerticalRectification | None = None
        rc = cfg["rectification"]
        if rc:
            lens = [(self.dist.to_corrected(a), self.dist.to_corrected(b)) for a, b in rc["lines"]]
            self.rect = VerticalRectification(vertical_vanishing_point(lens, size), rc["focal_px"], size)
        cal = cfg["calibration"]
        focal = cal["focal_px"] if cal else (self.rect.focal_px if self.rect is not None else None)
        side_a, side_b = resolve_sides(cfg["long_side_m"], cfg["short_side_m"], cfg["first_side"])
        self.projection: CameraProjection = compute_projection(
            cfg["points"], side_a, side_b, size, focal_px=focal,
            vertical=self.rect.vertical if self.rect is not None else None)

    def distance_m(self, xy, corrected: bool = False) -> float:
        """corrected=True: xy は魚眼補正済みのフレームの座標（抽出時に補正した結果）なので歪み補正はしない。"""
        q = self.dist.to_corrected(xy) if not corrected else xy
        if self.rect is not None:
            q = self.rect.to_rectified(q)
        return self.projection.distance_m((float(q[0]), float(q[1])))


class CameraDistance:
    """カメラ設定フォルダ内の <カメラID>.json から、カメラIDと足元の座標の組をカメラからの距離(m)にする。"""

    def __init__(self, config_dir: Path | str, image_dir: Path | str | None = None):
        self.config_dir = Path(config_dir)
        self.image_dir = Path(image_dir) if image_dir is not None and str(image_dir).strip() else self.config_dir
        self._configs = self._index(self.config_dir)
        self._images = self._index(self.image_dir)
        self._cameras: dict[str, _Camera | None] = {}
        self.errors: dict[str, str] = {}     # カメラID -> 距離を出せなかった理由

    @staticmethod
    def _index(folder: Path) -> dict[str, Path]:
        """フォルダ直下のファイルを「拡張子|ファイル名（拡張子なし）」（小文字）で引けるようにする。"""
        files: dict[str, Path] = {}
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file():
                    files.setdefault(f"{p.suffix.lower()}|{p.stem.lower()}", p)
        return files

    def _image_size(self, camera_id: str) -> tuple[int, int] | None:
        """カメラ画像（<カメラID>.png など）の画像サイズ (幅, 高さ)。"""
        for s in sorted(IMAGE_SUFFIXES):
            path = self._images.get(f"{s}|{camera_id.lower()}")
            img = imread(path) if path is not None else None
            if img is not None:
                return img.shape[1], img.shape[0]
        return None

    def _camera(self, camera_id: str) -> _Camera | None:
        key = camera_id.lower()
        if key not in self._cameras:
            cam, err = None, ""
            config = self._configs.get(f".json|{camera_id.lower()}")
            if config is None:
                err = f"設定ファイル {camera_id}.json が見つかりません"
            else:
                cfg = load_config(config)
                size = self._image_size(camera_id)
                if cfg is None:
                    err = f"{config.name} を読み込めません"
                elif size is None:
                    err = f"画像サイズを取得するカメラ画像（{self.image_dir}/{camera_id}.png）が見つかりません"
                else:
                    try:
                        cam = _Camera(cfg, size)
                    except (ProjectionError, ValueError) as e:
                        err = f"{config.name} から射影変換を作れません: {e}"
            self._cameras[key] = cam
            if err:
                self.errors[camera_id] = err
        return self._cameras[key]

    def distance_m(self, camera_id: str, foot_xy, corrected: bool = False) -> float | None:
        """カメラ直下の地面点から足元までの地面上の距離(m)。算出できなければ None。
        corrected=True: foot_xy は魚眼補正済みのフレームの座標（抽出時に補正した結果）。歪み補正を省く（垂直補正は従来どおり）。"""
        if not foot_xy or len(foot_xy) < 2:
            return None
        cam = self._camera(camera_id)
        if cam is None:
            return None
        try:
            d = cam.distance_m((float(foot_xy[0]), float(foot_xy[1])), corrected)
        except (ValueError, ZeroDivisionError):
            return None
        return d if d == d and abs(d) != float("inf") else None
