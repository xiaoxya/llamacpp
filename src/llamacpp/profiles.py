"""Profile 多配置管理。

存储：~/.config/llamacpp/profiles/<名字>.env（server.env 同格式）；
激活：profile use 将其渲染写入 server.env，并记录 .active 指针。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import (
    ServerConfig,
    atomic_write,
    parse_env_file,
    render_server_config,
    save_server_config,
)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProfileError(Exception):
    pass


def profiles_dir(config_dir: Path) -> Path:
    return config_dir / "profiles"


def active_file(config_dir: Path) -> Path:
    return profiles_dir(config_dir) / ".active"


def validate_name(name: str) -> str:
    if not NAME_RE.fullmatch(name or ""):
        raise ProfileError(f"非法 profile 名：{name!r}（只允许字母数字 ._-,且以字母数字开头）")
    return name


def list_profiles(config_dir: Path) -> list[Path]:
    directory = profiles_dir(config_dir)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.env") if p.is_file())


def active_profile(config_dir: Path) -> str | None:
    path = active_file(config_dir)
    try:
        name = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    return name or None


def profile_path(config_dir: Path, name: str) -> Path:
    validate_name(name)
    return profiles_dir(config_dir) / f"{name}.env"


def create_profile(
    config_dir: Path,
    name: str,
    server_env: Path,
    description: str = "",
) -> Path:
    """把当前生效的 server.env 快照为新 profile。"""
    target = profile_path(config_dir, name)
    if target.exists():
        raise ProfileError(f"profile 已存在:{name}")
    values, warnings = parse_env_file(server_env, "server")
    cfg = ServerConfig(**values)
    errors = cfg.validate()
    if errors:
        raise ProfileError("当前配置无效，拒绝快照：" + "；".join(errors))
    header = f"# llamacpp profile: {name}\n"
    if description:
        header += f"# {description}\n"
    content = header + render_server_config(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, content)
    return target


def use_profile(
    config_dir: Path,
    name: str,
    server_env: Path,
    dry_run: bool = False,
) -> ServerConfig:
    """激活 profile：校验后写入 server.env 并更新 .active 指针。"""
    source = profile_path(config_dir, name)
    if not source.exists():
        raise ProfileError(f"profile 不存在:{name}")
    values, _warnings = parse_env_file(source, "server")
    cfg = ServerConfig(**values)
    save_server_config(cfg, server_env, dry_run=dry_run)
    if not dry_run:
        active_file(config_dir).parent.mkdir(parents=True, exist_ok=True)
        atomic_write(active_file(config_dir), f"{name}\n", mode=0o644)
    return cfg


def delete_profile(config_dir: Path, name: str) -> None:
    source = profile_path(config_dir, name)
    if not source.exists():
        raise ProfileError(f"profile 不存在:{name}")
    source.unlink()
    if active_profile(config_dir) == name:
        active_file(config_dir).unlink(missing_ok=True)


def show_profile(config_dir: Path, name: str) -> str:
    source = profile_path(config_dir, name)
    if not source.exists():
        raise ProfileError(f"profile 不存在:{name}")
    return source.read_text(encoding="utf-8")


def import_legacy_snapshot(config_dir: Path, name: str, source_env: Path) -> Path:
    """从任意 env 文件导入为 profile（不做有效性强校验之外的转换）。"""
    target = profile_path(config_dir, name)
    if target.exists():
        raise ProfileError(f"profile 已存在:{name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_env, target)
    return target
