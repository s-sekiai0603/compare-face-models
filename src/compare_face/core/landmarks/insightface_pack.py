"""InsightFace のモデルパック buffalo_l（GitHub Releases 配布, 約 290MB）から必要な ONNX を取り出す。

buffalo_l.zip には det_10g.onnx（SCRFD 顔検出 + 5 点）, 2d106det.onnx（106 点ランドマーク）,
w600k_r50.onnx, 1k3d68.onnx, genderage.onnx が入っている。必要なファイルだけ models/landmark/ に展開し、zip は削除する。
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from ..common.weights import LANDMARK_DIR

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
PACK_FILES = ("det_10g.onnx", "2d106det.onnx")  # 本ツールが使うファイル。zip 取得時にまとめて展開する


def ensure_buffalo_file(filename: str, log: Callable[[str], None] = print, dst_dir: Path = LANDMARK_DIR) -> Path:
    """dst_dir/filename が無ければ buffalo_l.zip を取得して展開する。"""
    dst = dst_dir / filename
    if dst.is_file():
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / "buffalo_l.zip.part"
    log(f"InsightFace buffalo_l（約 290MB）をダウンロードしています: {BUFFALO_L_URL}")
    try:
        urllib.request.urlretrieve(BUFFALO_L_URL, str(tmp))
        with zipfile.ZipFile(tmp) as z:
            names = set(z.namelist())
            for name in {filename, *PACK_FILES} & names:
                if not (dst_dir / name).is_file():
                    z.extract(name, str(dst_dir))
                    log(f"展開: {dst_dir / name}")
    finally:
        tmp.unlink(missing_ok=True)
    if not dst.is_file():
        raise RuntimeError(f"{filename} を buffalo_l.zip から取り出せませんでした")
    return dst
