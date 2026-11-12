"""距離・向き・精度分析で出力したチャートの PNG を、条件を選んで重ねて見るための処理（GUI 非依存。画面は ui/chart_viewer_screen.py）。

重ねる対象は「色・アイコンの条件 = 組合せごと」で出力した PNG（1 ファイル = 1 カメラ表記 × 1 色 × 1 アイコン）。
ファイル名は distance_pose_chart.chart_png_path の規則 `<Excel名>_チャート_<カメラ>・<色>・<アイコン>.png`（空白などは _ に置換）。
OR 条件で出力した PNG など、この名前に当てはまらないファイルは対象外。
組合せごとの出力は全 PNG で軸の範囲が同じなので、そのまま重ねられる。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .distance_pose_chart import COLOR_GROUPS, PLOT, icon_choices, safe_label

CHART_MARK = "_チャート_"
ALL_CAMERAS = "すべてのカメラ"
CAMERA_PREFIX = "カメラ_"      # 「カメラ <ID>」のラベルを safe_label にかけた形（カメラ_<ID>）
TITLE_BAND_RATIO = PLOT[1] * 0.9   # チャート上端からプロット領域の手前まで（タイトルの帯）の高さの割合。重ねる 2 枚目以降は消す

# 色・アイコンの表記（ファイル名の形）→ 段階の番号 / アイコンのファイル名
_COLOR_OF_NAME = {safe_label(label): g for g, (label, _, _) in enumerate(COLOR_GROUPS)}
_ICON_OF_NAME = {safe_label(label): fname for label, fname in icon_choices()}

Key = tuple[str, int, str]   # (カメラ表記, 色の段階 g, アイコンのファイル名)


@dataclass(frozen=True)
class ChartPngIndex:
    """フォルダにある組合せごとの PNG の一覧。"""
    files: dict[str, dict[Key, Path]]   # Excel 名 → {(カメラ表記, g, アイコン): パス}

    def excel_names(self) -> list[str]:
        return sorted(self.files)

    def cameras(self, excel_name: str) -> list[str]:
        """カメラ表記（カメラ_<ID>。「すべてのカメラ」は含めない）の昇順。"""
        return sorted({k[0] for k in self.files.get(excel_name, {}) if k[0] != ALL_CAMERAS})

    def has_all_cameras(self, excel_name: str) -> bool:
        return any(k[0] == ALL_CAMERAS for k in self.files.get(excel_name, {}))


def camera_display(camera: str) -> str:
    """カメラ表記 → 画面に出す名前（カメラ_Cam1 → Cam1）。"""
    return camera[len(CAMERA_PREFIX):] if camera.startswith(CAMERA_PREFIX) else camera


def parse_chart_png_name(name: str) -> tuple[str, Key] | None:
    """PNG のファイル名 → (Excel 名, (カメラ表記, 色の段階, アイコン))。組合せごとの名前でなければ None。"""
    if not name.lower().endswith(".png") or CHART_MARK not in name:
        return None
    excel_name, _, tail = name[:-4].partition(CHART_MARK)
    parts = tail.rsplit("・", 2)
    if not excel_name or len(parts) != 3:
        return None
    camera, color, icon = parts
    g, fname = _COLOR_OF_NAME.get(color), _ICON_OF_NAME.get(icon)
    if g is None or fname is None or not camera:
        return None
    return excel_name, (camera, g, fname)


def scan_chart_pngs(folder: Path | str) -> ChartPngIndex:
    """フォルダ直下の組合せごとの PNG を Excel 名ごとに集める。フォルダが無ければ空。"""
    files: dict[str, dict[Key, Path]] = {}
    root = Path(folder)
    if root.is_dir():
        for p in sorted(root.iterdir()):
            parsed = parse_chart_png_name(p.name) if p.is_file() else None
            if parsed is not None:
                files.setdefault(parsed[0], {})[parsed[1]] = p
    return ChartPngIndex(files)


def select_paths(index: ChartPngIndex, excel_name: str, cameras: list[str], colors: list[int], icons: list[str]) -> tuple[list[Path], int]:
    """条件に合う PNG と、条件に合うのに見つからなかった組合せ数（点が無くて出力されなかったもの）。
    cameras は 1 つ以上のカメラ表記（すべてのカメラ = [ALL_CAMERAS]、カメラ個別 = [カメラ_<ID>]）。
    colors / icons は手前に出したい順（先頭ほど手前）。戻り値の PNG は重ねる順（奥 → 手前）で、
    色の順を優先し（色が手前のものほど後）、同じ色の中ではアイコンの順に重なる。"""
    table = index.files.get(excel_name, {})
    found, missing = [], 0
    for cam in cameras:
        for g in reversed(list(colors)):
            for fname in reversed(list(icons)):
                p = table.get((cam, g, fname))
                if p is None:
                    missing += 1
                else:
                    found.append(p)
    return found, missing


def load_rgba(path: Path | str):
    """PNG を RGBA の Pillow 画像で読む（日本語パスでも読める）。"""
    import io
    from PIL import Image
    return Image.open(io.BytesIO(Path(path).read_bytes())).convert("RGBA")


def overlay_png_path(out_dir: Path | str, excel_name: str, camera: str, now: datetime | None = None) -> Path:
    """重ねた画像の出力先ファイル名 `<Excel名>_重ね合わせ_<カメラ>_<日時>.png`（out_dir の中。カメラはすべてのカメラ / カメラ_<ID>）。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    parts = [excel_name or "チャート", "重ね合わせ", safe_label(camera) or "カメラ", stamp]
    return Path(out_dir) / ("_".join(parts) + ".png")


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


def save_overlay_png(image, out_dir: Path | str, excel_name: str, camera: str, now: datetime | None = None) -> Path:
    """重ねた画像を既定のファイル名（overlay_png_path）で out_dir に PNG 保存して、そのパスを返す。"""
    return write_png(image, overlay_png_path(out_dir, excel_name, camera, now))


class ChartOverlay:
    """PNG を重ねた画像（RGBA）を作る。選択に足した分だけを前回の結果へ重ねて、読み込み・合成の量を減らす。
    先頭の PNG はそのまま、2 枚目以降はタイトルの帯を消してから重ねる（タイトルが何重にもならない）。"""

    def __init__(self, loader=load_rgba):
        self._loader = loader
        self._paths: list[Path] = []
        self._image = None

    @property
    def image(self):
        """直近に重ねた画像（なければ None）。次の compose で書き換わるので、保存は処理が終わっているときに行う。"""
        return self._image

    def compose(self, paths: list[Path], progress=None):
        """paths を重ねた画像。空なら None。progress(済んだ枚数, 今回重ねる枚数) は 1 枚重ねるごとに呼ぶ（画面の進捗表示用）。"""
        from PIL import Image
        wanted = [Path(p) for p in paths]
        if not wanted:
            self._paths, self._image = [], None
            return None
        if self._image is None or wanted[:len(self._paths)] != self._paths:
            self._paths, self._image = [], None      # 外した・順序が変わった → 最初から重ね直す（手前に足しただけなら続きから）
        todo = wanted[len(self._paths):]
        if progress is not None:
            progress(0, len(todo))
        for n, p in enumerate(todo, 1):
            im = self._loader(p)
            if self._image is None:
                self._image = im.copy()
            else:
                if im.size != self._image.size:
                    im = im.resize(self._image.size, Image.LANCZOS)
                band = int(im.height * TITLE_BAND_RATIO)
                im.paste((0, 0, 0, 0), (0, 0, im.width, band))
                self._image.alpha_composite(im)
            self._paths.append(p)
            if progress is not None:
                progress(n, len(todo))
        return self._image
