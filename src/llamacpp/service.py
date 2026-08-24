"""systemd user service 管理：unit 文件生成、启停、日志。

unit 内容与 Bash 版 write_service 保持一致。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SERVICE_UNIT_TEMPLATE = """\
[Unit]
Description=llama.cpp OpenAI/Anthropic API server (dual NVIDIA GPU)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/bin/{launcher_name}
Restart=on-failure
RestartSec=10
TimeoutStopSec=45
KillSignal=SIGTERM
Environment=PATH=/opt/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/bin
Environment=LD_LIBRARY_PATH=/opt/cuda/lib64

[Install]
WantedBy=default.target
"""

LAUNCHER_TEMPLATE = """\
#!/usr/bin/env bash
set -Eeuo pipefail
exec "${{HOME}}/.local/bin/{manager_name}" run-server "$@"
"""


def render_unit(launcher_name: str = "llamacpp-start") -> str:
    return SERVICE_UNIT_TEMPLATE.format(launcher_name=launcher_name)


def render_launcher(manager_name: str = "llamacpp") -> str:
    return LAUNCHER_TEMPLATE.format(manager_name=manager_name)


def systemctl_available() -> bool:
    return shutil.which("systemctl") is not None


def _systemctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def service_file_exists(service_path: Path) -> bool:
    return systemctl_available() and service_path.exists()


def is_active(service_name: str) -> bool:
    if not systemctl_available():
        return False
    result = _systemctl("is-active", "--quiet", service_name)
    return result.returncode == 0


def daemon_reload(dry_run: bool = False) -> None:
    if dry_run:
        print("[DRY-RUN] systemctl --user daemon-reload")
        return
    if systemctl_available():
        _systemctl("daemon-reload")


def enable(service_name: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] systemctl --user enable {service_name}")
        return
    _systemctl("enable", service_name)


def disable(service_name: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] systemctl --user disable {service_name}")
        return
    _systemctl("disable", service_name)


def start(service_name: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] systemctl --user start {service_name}")
        return
    _systemctl("start", service_name)


def stop(service_name: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] systemctl --user stop {service_name}")
        return
    _systemctl("stop", service_name)


def status_text(service_name: str) -> str:
    result = _systemctl("status", service_name, "--no-pager")
    return result.stdout or result.stderr


def journalctl_lines(service_name: str, lines: int, follow: bool) -> list[str]:
    args = ["journalctl", "--user", "-u", service_name, "-n", str(lines), "--no-pager"]
    if follow:
        args.append("-f")
    result = subprocess.run(args, check=False)
    return [str(result.returncode)]


def is_service_active_quiet(conflicting: str) -> bool:
    return is_active(conflicting)
