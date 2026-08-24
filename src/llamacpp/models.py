"""GGUF 模型扫描与选择。分片模型只保留 -00001-of-XXXXX 首片。"""

from __future__ import annotations

import re
from pathlib import Path

SHARD_RE = re.compile(r"-([0-9]{5})-of-[0-9]{5}\.gguf$", re.IGNORECASE)


def scan_models(root: Path) -> list[Path]:
    """递归扫描 root 下的 GGUF；排除 mmproj 与非首分片，返回去重排序列表。"""
    root = Path(root)
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file() or not path.name.lower().endswith(".gguf"):
            continue
        if path.name.lower().startswith("mmproj"):
            continue
        match = SHARD_RE.search(path.name)
        if match and match.group(1) != "00001":
            continue
        found.add(path)
    return sorted(found)


def human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}T"


def choose_model(models: list[Path], prompt=input) -> Path | None:
    """交互选择模型；返回所选路径或 None（取消）。非交互调用方自行处理。"""
    if not models:
        return None
    print("\n检测到以下 GGUF 模型：")
    for i, model in enumerate(models, start=1):
        try:
            size = human_size(model)
        except OSError:
            size = "?"
        print(f"  {i}) [{size}] {model}")
    while True:
        answer = prompt(f"请选择模型 [1-{len(models)}，q 取消]: ").strip()
        if answer == "q":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]
        print("选择无效。")
