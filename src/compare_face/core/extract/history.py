"""画像抽出の履歴ファイル（出力先/<日時>/抽出履歴.json）と、魚眼補正の有無の判定。

魚眼補正して抽出した結果の座標（bbox・骨格・足元）は補正後の画像の座標。後段（距離の算出・足元ヒートマップ等）は
追跡IDフォルダから履歴ファイルを辿って「補正済みか」「補正の λ」を知り、補正済みなら再補正しない。
ファイル名は日時フォルダ名と同じにしない（追跡IDフォルダの走査が「フォルダ名 = json の名前」で拾うため）。"""
from __future__ import annotations

import json
from pathlib import Path

from ..camera.projective import load_config

HISTORY_FILE_NAME = "抽出履歴.json"


class FisheyeConfigError(Exception):
    """魚眼補正に必要なカメラ設定が揃っていない。"""


def find_camera_config(camera_dir: Path, camera_id: str) -> Path | None:
    """<カメラID>.json（大文字小文字は無視）。"""
    want = f"{camera_id}.json".lower()
    try:
        for p in camera_dir.iterdir():
            if p.is_file() and p.name.lower() == want:
                return p
    except OSError:
        pass
    return None


def load_fisheye_lambdas(camera_dir: Path | str | None, camera_ids) -> dict[str, float]:
    """カメラIDごとの歪み係数 λ（設定ファイルの distortion）。設定ファイルが無い・読めない・魚眼補正の情報が無いカメラがあれば
    FisheyeConfigError（全カメラ分の問題をまとめて報告する）。"""
    if not camera_dir or not Path(camera_dir).is_dir():
        raise FisheyeConfigError(f"カメラ設定フォルダが見つかりません: {camera_dir or '（未指定）'}")
    folder = Path(camera_dir)
    lams: dict[str, float] = {}
    problems: list[str] = []
    for cam in sorted(set(camera_ids)):
        path = find_camera_config(folder, cam)
        if path is None:
            problems.append(f"{cam}: 設定ファイル（{cam}.json）が見つかりません")
            continue
        cfg = load_config(path)
        if cfg is None:
            problems.append(f"{cam}: 設定ファイル（{path.name}）を読み込めません")
        elif cfg["distortion"] is None:
            problems.append(f"{cam}: 設定ファイル（{path.name}）に魚眼補正の情報（歪み補正）がありません")
        else:
            lams[cam] = float(cfg["distortion"]["lambda"])
    if problems:
        raise FisheyeConfigError("魚眼補正に必要なカメラ設定を取得できません。\n" + "\n".join(problems))
    return lams


def write_history(run_dir: Path, info: dict) -> Path:
    path = run_dir / HISTORY_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def fisheye_lambda_of_track(track_dir: Path) -> float | None:
    """追跡IDフォルダ（…/<日時>/<カメラID>/<映像名>/<追跡ID>）の結果が魚眼補正済みなら補正の λ、そうでなければ None。
    日時フォルダの履歴ファイルを優先し、無ければ追跡ID json の fisheye_corrected / fisheye_lambda を見る。
    どちらも無い（古い抽出結果）は未補正。"""
    track_dir = Path(track_dir)
    parents = track_dir.parents
    if len(parents) >= 3:
        hist = _read_json(parents[2] / HISTORY_FILE_NAME)
        if hist is not None and "fisheye_correct" in hist:
            if not hist["fisheye_correct"]:
                return None
            lam = (hist.get("fisheye_lambdas") or {}).get(parents[1].name)
            if lam is None:  # カメラIDの大文字小文字の違いを許す
                lam = next((v for k, v in (hist.get("fisheye_lambdas") or {}).items()
                            if k.lower() == parents[1].name.lower()), None)
            if lam is not None:
                return float(lam)
    data = _read_json(track_dir / f"{track_dir.name}.json")
    if data and data.get("fisheye_corrected") and data.get("fisheye_lambda") is not None:
        return float(data["fisheye_lambda"])
    return None
