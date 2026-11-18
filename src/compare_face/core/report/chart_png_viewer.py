"""距離・向き・精度分析で出力したチャートの PNG を、条件を選んで重ねて見るための処理（GUI 非依存。画面は ui/chart_viewer_screen.py）。

重ねる対象は「色・アイコンの条件 = 組合せごと」で出力した PNG（1 ファイル = 1 カメラ表記 × 1 色 × 1 アイコン）。
ファイル名は distance_pose_chart.chart_png_path の規則 `<Excel名>_チャート_<カメラ>・<色>・<アイコン>.png`（空白などは _ に置換）。
OR 条件で出力した PNG など、この名前に当てはまらないファイルは対象外。
組合せごとの出力は全 PNG で軸の範囲が同じなので、そのまま重ねられる。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .distance_pose_chart import COLOR_GROUPS, ICON_PX, PLOT, color_group_rgb, icon_choices, safe_label

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


def save_overlay_png(image, out_dir: Path | str, excel_name: str, camera: str, now: datetime | None = None) -> Path:
    """重ねた画像を既定のファイル名（overlay_png_path）で out_dir に PNG 保存して、そのパスを返す。"""
    return write_png(image, overlay_png_path(out_dir, excel_name, camera, now))


HEATMAP_CACHE_DIR = ".heatmap_cache"      # PNG の格納先の中に、作成したヒートマップの PNG を保存するフォルダ（scan_chart_pngs は直下のファイルだけ見るので対象外）
HEAT_DOWNSCALE = 4          # ヒートマップ計算時の縮小率（プロット領域を 1/4 にして密度を求め、最後に拡大する）
HEAT_SIGMA_ICONS = 1.6      # ぼかしの大きさ（アイコン幅に対する倍率）。点の密集具合をなめらかにする
HEAT_MAX_ALPHA = 0.92
HEAT_ALPHA_GAMMA = 0.45     # 小さいほど、点が少ない所も濃く出る
_MIN_SATURATION = 40        # アイコンの色づいた部分の判定（max - min）。グリッド線・文字は無彩色なので入らない


def _icon_mask(im):
    """チャート PNG のプロット領域の中で、アイコンの色づいた部分の真偽配列と、その領域の座標 (x0, y0, x1, y1)。
    グリッド線・文字・枠は無彩色なので除かれる。重なって背面に隠れたアイコンは、PNG が 1 色 × 1 アイコンごとに別なので全て拾える。"""
    import numpy as np
    a = np.asarray(im)
    h, w = a.shape[:2]
    px, py, pw, ph = PLOT
    x0, y0, x1, y1 = int(px * w), int(py * h), int((px + pw) * w), int((py + ph) * h)
    c = a[y0:y1, x0:x1].astype(np.int16)
    sat = c[..., :3].max(axis=-1) - c[..., :3].min(axis=-1)
    return (c[..., 3] >= 128) & (sat >= _MIN_SATURATION), (x0, y0, x1, y1)


def _heat_palette(levels: "np.ndarray"):
    """色の段階（0〜6 の連続値）→ RGB（float32, N×3）。段階ごとの色（緑 → 黄 → 橙 → 赤 → 濃赤）の間を補間する。"""
    import numpy as np
    stops = np.array([color_group_rgb(g) for g in range(len(COLOR_GROUPS))], np.float32)
    t = np.clip(levels, 0, len(stops) - 1)
    k = np.minimum(t.astype(np.int32), len(stops) - 2)
    f = (t - k)[..., None]
    return stops[k] * (1 - f) + stops[k + 1] * f


class HeatmapOverlay:
    """PNG 内のアイコンの位置・色から、アイコンを消したヒートマップ画像（RGBA）を作る（ChartOverlay と同じ使い方）。
    PNG は 1 色 × 1 アイコンごとに別ファイルなので、重ねたときに背面で見えなくなるアイコンも全て数える。
    色 = 周辺の点の色の段階の平均（緑 → 赤）、濃さ = 点の密集具合。軸・グリッド・凡例は最初の PNG のものを使う。"""

    def __init__(self, loader=load_rgba, cache_dir: Path | str | None = None):
        self._loader = loader
        self.cache_dir = Path(cache_dir) if cache_dir else None    # 作成したヒートマップの PNG の保存先（None = 保存しない）
        self._cache: dict[Path, tuple] = {}   # パス → (縮小したアイコンの割合, 色の段階, プロット領域の座標)
        self._image = None
        self._frame_path: Path | None = None
        self._frame = None

    @property
    def image(self):
        return self._image

    def _level(self, path: Path) -> int:
        parsed = parse_chart_png_name(path.name)
        return parsed[1][1] if parsed else 0

    def cache_path(self, paths: list[Path]) -> Path | None:
        """この PNG の組合せ（順序は関係ない）のヒートマップを保存するパス。PNG の更新日時・サイズ・ヒートマップの設定が変わると別のパスになる。"""
        import hashlib
        if self.cache_dir is None:
            return None
        parts = [f"v1|{HEAT_DOWNSCALE}|{HEAT_SIGMA_ICONS}|{HEAT_MAX_ALPHA}|{HEAT_ALPHA_GAMMA}|{_MIN_SATURATION}"]
        for p in sorted(Path(x) for x in paths):
            st = p.stat()
            parts.append(f"{p.name}|{st.st_size}|{st.st_mtime_ns}")
        return self.cache_dir / (hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest() + ".png")

    def compose(self, paths: list[Path], progress=None):
        """ヒートマップ画像。同じ条件で作った保存済みの PNG があればそれを読む。なければ作って保存する。"""
        wanted = [Path(p) for p in paths]
        if not wanted:
            self._image = None
            return None
        saved = self.cache_path(wanted)
        if saved is not None and saved.is_file():
            try:
                self._image = self._loader(saved)
                if progress is not None:
                    progress(len(wanted), len(wanted))
                return self._image
            except Exception:
                pass      # 壊れていれば作り直す
        self._image = self._build(wanted, progress)
        if saved is not None and self._image is not None:
            try:
                write_png(self._image, saved)
            except OSError:
                pass      # 保存できなくても表示はできる
        return self._image

    def _build(self, wanted: list[Path], progress=None):
        import cv2
        import numpy as np
        from PIL import Image
        if progress is not None:
            progress(0, len(wanted))
        density = weighted = None
        for n, p in enumerate(wanted, 1):
            if p not in self._cache:
                im = self._loader(p)
                mask, box = _icon_mask(im)
                frac = cv2.resize(mask.astype(np.float32), (max(mask.shape[1] // HEAT_DOWNSCALE, 1), max(mask.shape[0] // HEAT_DOWNSCALE, 1)),
                                  interpolation=cv2.INTER_AREA)
                self._cache[p] = (frac, self._level(p), box)
            frac, level, box = self._cache[p]
            if density is None:
                density, weighted, first_box = np.zeros_like(frac), np.zeros_like(frac), box
            if frac.shape != density.shape:
                frac = cv2.resize(frac, (density.shape[1], density.shape[0]), interpolation=cv2.INTER_AREA)
            density += frac
            weighted += frac * level
            if progress is not None:
                progress(n, len(wanted))
        # 枠（軸・グリッド・凡例）: 最初の PNG からプロット領域内のアイコンを消したもの
        if self._frame is None or self._frame_path != wanted[0]:
            im0 = self._loader(wanted[0])
            mask, (fx0, fy0, fx1, fy1) = _icon_mask(im0)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            arr = np.array(im0)
            region = arr[fy0:fy1, fx0:fx1]
            region[mask] = 0
            self._frame, self._frame_path = Image.fromarray(arr, "RGBA"), wanted[0]
        frame = self._frame
        x0, y0, x1, y1 = first_box
        sigma = max(HEAT_SIGMA_ICONS * ICON_PX / HEAT_DOWNSCALE, 1.0)
        d = cv2.GaussianBlur(density, (0, 0), sigma)
        w = cv2.GaussianBlur(weighted, (0, 0), sigma)
        peak = float(np.percentile(d[d > 1e-6], 99)) if (d > 1e-6).any() else 0.0
        if peak <= 0:
            return frame.copy()
        mean_level = np.where(d > 1e-6, w / np.maximum(d, 1e-6), 0)
        alpha = np.clip(d / peak, 0, 1) ** HEAT_ALPHA_GAMMA * HEAT_MAX_ALPHA
        alpha[d < peak * 1e-3] = 0
        rgb = _heat_palette(mean_level)
        heat = np.dstack([rgb, alpha * 255]).astype(np.float32)
        heat = cv2.resize(heat, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((frame.height, frame.width, 4), np.uint8)
        canvas[y0:y1, x0:x1] = np.clip(heat, 0, 255).astype(np.uint8)
        out = Image.fromarray(canvas, "RGBA")
        out.alpha_composite(frame)
        return out


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
