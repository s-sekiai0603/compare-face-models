"""CVLFace（Hugging Face の minchul/cvlface_*）のリポジトリ取得と直接読み込み。

モデルカードの例は transformers の AutoModel で読むが、同梱ラッパーがカレントディレクトリ依存の
相対パスで重みを開くため環境によって失敗する。ここでは transformers を使わず、取得したリポジトリの
models / aligners パッケージを一意な名前で import し、get_model / get_aligner を直接呼ぶ。
リポジトリは特徴量モデルなら models/feature/cvlface/<リポジトリ名>/、アライナなら models/landmark/<リポジトリ名>/ に置く。
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Callable

from .weights import FEATURE_DIR

CVLFACE_FEATURE_DIR = FEATURE_DIR / "cvlface"
HF_URL = "https://huggingface.co/"
IGNORE_PATTERNS = ("model.safetensors", "*.png", "*.jpg")  # 重みは pretrained_model/model.pt を使う


def local_dir(repo_id: str, base_dir: Path = CVLFACE_FEATURE_DIR) -> Path:
    return base_dir / repo_id.split("/")[-1]


def is_downloaded(repo_id: str, package: str, base_dir: Path = CVLFACE_FEATURE_DIR) -> bool:
    d = local_dir(repo_id, base_dir)
    return ((d / "pretrained_model" / "model.pt").is_file()
            and (d / "pretrained_model" / "model.yaml").is_file()
            and (d / package / "__init__.py").is_file())


def manual_instructions(name: str, repo_id: str, base_dir: Path = CVLFACE_FEATURE_DIR) -> str:
    d = local_dir(repo_id, base_dir)
    return (
        f"{name} のモデルが見つかりません。画面を開いたときに自動取得します（ネットワーク接続が必要）。\n"
        f"手動で用意する場合は {HF_URL}{repo_id} のファイル一式（model.safetensors 以外）を次の場所に配置してください。\n"
        f"   {d}\n"
        f"コマンド: python -m compare_face download  （登録済みモデルをまとめて取得）"
    )


def download_repo(repo_id: str, log: Callable[[str], None] = print, base_dir: Path = CVLFACE_FEATURE_DIR) -> Path:
    """Hugging Face からリポジトリ一式を取得する（取得済みファイルはスキップされる）。"""
    from huggingface_hub import snapshot_download
    d = local_dir(repo_id, base_dir)
    d.mkdir(parents=True, exist_ok=True)
    log(f"{repo_id} をダウンロードしています: {d}")
    snapshot_download(repo_id, local_dir=str(d), ignore_patterns=list(IGNORE_PATTERNS))
    return d


def import_package(pkg_dir: Path, unique_name: str) -> types.ModuleType:
    """pkg_dir にあるパッケージを unique_name で import する（内部の相対 import はそのまま動く）。
    複数のリポジトリが同名パッケージ（models / aligners）を持つため、名前を分けて衝突を避ける。"""
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    init = pkg_dir / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"パッケージが見つかりません: {init}")
    spec = importlib.util.spec_from_file_location(unique_name, str(init),
                                                  submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    # KP-RPE 同梱の RPE/__init__.py は rpe_ops のビルドのために os.chdir() し、ビルドを止めると元に戻さない。
    # 以後の相対パスが壊れないよう、import 前のカレントディレクトリを必ず復元する。
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(unique_name, None)
        raise
    finally:
        os.chdir(cwd)
    return module


def load_conf(yaml_path: Path):
    """model.yaml を OmegaConf（属性アクセスできる設定）として読む。"""
    import yaml
    from omegaconf import OmegaConf
    return OmegaConf.create(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))


@contextlib.contextmanager
def without_extension_build():
    """KP-RPE の RPE パッケージは CUDA 拡張 rpe_ops が無いと setup.py でのビルドを試み、
    成功すると sys.exit() でプロセスを終了してしまう。ビルドは行わず、純 PyTorch 実装の
    フォールバック（結果は同じ・やや低速）を使うよう、import 中だけビルド呼び出しを失敗させる。"""
    original = subprocess.check_call

    def _refuse(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0] if args else "setup.py")

    subprocess.check_call = _refuse
    try:
        yield
    finally:
        subprocess.check_call = original


def torch_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def describe_device() -> str:
    import torch
    return (f"GPU (cuda:0, {torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU")
