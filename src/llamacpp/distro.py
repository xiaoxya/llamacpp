"""发行版检测与包管理器抽象：支持 Arch 系与 Debian/Ubuntu 系。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class UnsupportedDistroError(Exception):
    pass


@dataclass(frozen=True)
class DistroInfo:
    id: str          # 如 arch / manjaro / debian / ubuntu
    family: str      # arch / debian
    pretty_name: str

    @property
    def is_arch(self) -> bool:
        return self.family == "arch"

    @property
    def is_debian(self) -> bool:
        return self.family == "debian"


_ARCH_IDS = {"arch", "manjaro", "endeavouros", "cachyos", "garuda", "artix"}
_DEBIAN_FAMILY_IDS = {"debian", "ubuntu", "linuxmint", "pop", "kali", "raspbian"}


def parse_os_release(text: str) -> dict[str, str]:
    """解析 /etc/os-release 内容为字典（带引号自动去除）。"""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def detect_distro(os_release_path: Path | None = None) -> DistroInfo | None:
    path = os_release_path or Path("/etc/os-release")
    try:
        data = parse_os_release(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return None
    did = data.get("ID", "").lower()
    id_like = [x.lower() for x in data.get("ID_LIKE", "").split()]
    pretty = data.get("PRETTY_NAME") or did or "unknown"

    family = None
    if did in _ARCH_IDS or "arch" in id_like:
        family = "arch"
    elif did in _DEBIAN_FAMILY_IDS or "debian" in id_like:
        family = "debian"
    if family is None:
        return None
    return DistroInfo(id=did, family=family, pretty_name=pretty)


def current_distro() -> DistroInfo | None:
    return detect_distro()


# ------------------------------------------------------------ CUDA 路径 ----


def find_cuda_root(env: dict[str, str] | None = None) -> Path | None:
    """按优先级探测 CUDA Toolkit 根目录（含 bin/nvcc 才算有效）。

    顺序：CUDA_HOME → CUDA_PATH → /usr/local/cuda → /opt/cuda → PATH 中的 nvcc。
    覆盖 Ubuntu 官方 repo（/usr/local/cuda）、Arch cuda 包（/opt/cuda）
    与自定义安装。
    """
    env = env if env is not None else dict(os.environ)
    candidates: list[Path] = []
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if env.get(var):
            candidates.append(Path(env[var]))
    candidates += [Path("/usr/local/cuda"), Path("/opt/cuda")]
    for root in candidates:
        if (root / "bin" / "nvcc").is_file():
            return root
    which = shutil.which("nvcc", path=env.get("PATH", ""))
    if which:
        # <root>/bin/nvcc → root
        return Path(which).resolve().parent.parent
    return None


def nvcc_path(cuda_root: Path) -> Path:
    return cuda_root / "bin" / "nvcc"
