"""チャート確認画面の PNG 出力まわりの処理（GUI 非依存。画面は ui/chart_viewer_screen.py）。
表示しているチャートの画像を白い背景に重ね（on_white）、既定のファイル名（overlay_png_path）で PNG に保存する（write_png）。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .distance_pose_chart import safe_label

ALL_CAMERAS = "すべてのカメラ"
CAMERA_PREFIX = "カメラ_"      # 「カメラ <ID>」のラベルを safe_label にかけた形（カメラ_<ID>）


def camera_display(camera: str) -> str:
    """カメラ表記 → 画面に出す名前（カメラ_Cam1 → Cam1）。"""
    return camera[len(CAMERA_PREFIX):] if camera.startswith(CAMERA_PREFIX) else camera


def overlay_png_path(out_dir: Path | str, excel_name: str, camera: str, now: datetime | None = None) -> Path:
    """重ねた画像の出力先ファイル名 `<Excel名>_重ね合わせ_<カメラ>_<日時>.png`（out_dir の中。カメラはすべてのカメラ / カメラ_<ID>）。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    parts = [excel_name or "チャート", "重ね合わせ", safe_label(camera) or "カメラ", stamp]
    return Path(out_dir) / ("_".join(parts) + ".png")


def on_white(image):
    """透過した画像（RGBA）を白い背景に重ねた画像（RGBA）。元の画像は変えない。"""
    from PIL import Image
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.alpha_composite(image)
    return bg


def write_png(image, path: Path | str) -> Path:
    """画像（Pillow）を path に PNG で保存して、そのパスを返す（拡張子が .png でなければ付ける。フォルダが無ければ作る。日本語パス可）。"""
    import io
    path = Path(path)
    if path.suffix.lower() != ".png":
        path = path.with_name(path.name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    path.write_bytes(buf.getvalue())
    return path
