"""GPU 探测与查询，基于 nvidia-smi。"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query(fmt: str) -> list[list[str]]:
    if not nvidia_smi_available():
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fmt}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append([cell.strip() for cell in line.split(",")])
    return rows


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int
    temperature: int | None
    driver_version: str


def parse_gpu_table(output: str) -> list[GpuInfo]:
    """解析 nvidia-smi index,name,memory.total,memory.free,temperature.gpu,driver_version 输出。"""
    gpus: list[GpuInfo] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 4:
            continue

        def to_int(value: str) -> int | None:
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None

        gpus.append(
            GpuInfo(
                index=to_int(cells[0]) or 0,
                name=cells[1],
                memory_total_mib=to_int(cells[2]) or 0,
                memory_free_mib=to_int(cells[3]) or 0,
                temperature=to_int(cells[4]) if len(cells) > 4 else None,
                driver_version=cells[5] if len(cells) > 5 else "",
            )
        )
    return gpus


def list_gpus() -> list[GpuInfo]:
    if not nvidia_smi_available():
        return []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,temperature.gpu,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    return parse_gpu_table(proc.stdout)


def gpu_count() -> int:
    return len(_query("index"))


def low_memory_count(minimum_mib: int) -> int:
    totals = [int(row[0]) for row in _query("memory.total") if row[0].isdigit()]
    return sum(1 for mib in totals if mib < minimum_mib)


def detect_gpus(expected: int = 2, minimum_mib: int = 11000) -> bool:
    """要求指定数量的 GPU 且每卡显存达标。"""
    rows = _query("memory.total")
    totals = [int(r[0]) for r in rows if r[0].isdigit()]
    return len(totals) == expected and all(mib >= minimum_mib for mib in totals)


def gpu_models_are_mixed() -> bool:
    names = {row[0] for row in _query("name")}
    return len(names) > 1
