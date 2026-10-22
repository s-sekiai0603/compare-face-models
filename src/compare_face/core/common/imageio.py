"""日本語など非 ASCII を含むパスでも動く画像の読み書き。

Windows の cv2.imread / cv2.imwrite は非 ASCII パスを開けない（失敗しても例外を出さない）ため、
バイト列経由（np.fromfile + imdecode / imencode + tofile）で読み書きする。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """画像を読む。読めなければ None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """画像を書く（拡張子で形式を決める）。成功したら True。"""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img, params or [])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True
