"""路径常量，与 Bash 版保持一致。

所有函数在调用时读取环境变量，便于测试中通过 monkeypatch 重定向。
"""

from __future__ import annotations

import os
from pathlib import Path


def config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")


def user_bin_dir() -> Path:
    return Path.home() / ".local/bin"


def config_dir() -> Path:
    return config_home() / "llamacpp"


def server_config_file() -> Path:
    return config_dir() / "server.env"


def build_config_file() -> Path:
    return config_dir() / "build.env"


def legacy_config_file() -> Path:
    return config_dir() / "config.env"


def install_root() -> Path:
    return data_home() / "llamacpp"


def source_dir() -> Path:
    return install_root() / "src"


def build_dir() -> Path:
    return install_root() / "build"


def server_bin() -> Path:
    return build_dir() / "bin" / "llama-server"


def bench_bin() -> Path:
    return build_dir() / "bin" / "llama-bench"


def manager_bin() -> Path:
    return user_bin_dir() / "llamacpp"


def launcher_bin() -> Path:
    return user_bin_dir() / "llamacpp-start"


def systemd_user_dir() -> Path:
    return config_home() / "systemd" / "user"


def service_file(service_name: str = "llamacpp.service") -> Path:
    return systemd_user_dir() / service_name
