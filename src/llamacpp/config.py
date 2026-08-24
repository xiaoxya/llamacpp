"""配置读写与校验：server.env / build.env，格式与 Bash 版完全兼容。

约定：
- env 文件按纯数据解析（KEY=VALUE），绝不作为 shell 代码执行；
- 未知键告警并忽略；
- 旧键 LLAMA_CPP_REF 在非 server 类型中自动更名为 LLAMACPP_REF。
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# 与 Bash 版 SERVER_CONFIG_KEYS / BUILD_CONFIG_KEYS 顺序一致
SERVER_CONFIG_KEYS: tuple[str, ...] = (
    "MODEL_DIR", "MODEL", "MODEL_ALIAS", "MM_PROJ", "HOST", "PORT", "API_KEY",
    "CTX_SIZE", "N_PARALLEL", "BATCH_SIZE", "UBATCH_SIZE", "THREADS",
    "THREADS_BATCH", "N_GPU_LAYERS", "SPLIT_MODE", "TENSOR_SPLIT", "MAIN_GPU",
    "FIT", "FIT_TARGET", "FLASH_ATTN", "CACHE_TYPE_K", "CACHE_TYPE_V",
    "CONT_BATCHING", "CACHE_PROMPT", "JINJA", "REASONING", "REASONING_FORMAT",
    "REASONING_EFFORT", "REASONING_BUDGET", "METRICS", "WEBUI", "LOAD_MODE",
    "CUDA_VISIBLE_DEVICES", "EXTRA_ARGS",
)
BUILD_CONFIG_KEYS: tuple[str, ...] = ("CUDA_ARCHITECTURES", "BUILD_JOBS", "LLAMACPP_REF")
CONFIG_KEYS: tuple[str, ...] = SERVER_CONFIG_KEYS + BUILD_CONFIG_KEYS


class ConfigError(Exception):
    """配置解析或校验失败。"""


def expand_home(value: str) -> str:
    """展开行首的 ~ 或 ~/，与 Bash 版 expand_home 行为一致。"""
    home = str(Path.home())
    if value == "~":
        return home
    if value.startswith("~/"):
        return f"{home}/{value[2:]}"
    return value


class ServerConfig:
    """llama-server 启动参数；字段值保持字符串以贴近 Bash 版语义。"""

    DEFAULTS: dict[str, str] = {
        "MODEL_DIR": "~/models",
        "MODEL": "",
        "MODEL_ALIAS": "",
        "MM_PROJ": "",
        "HOST": "0.0.0.0",
        "PORT": "8080",
        "API_KEY": "",
        "CTX_SIZE": "32768",
        "N_PARALLEL": "2",
        "BATCH_SIZE": "512",
        "UBATCH_SIZE": "256",
        "THREADS": "8",
        "THREADS_BATCH": "16",
        "N_GPU_LAYERS": "all",
        "SPLIT_MODE": "layer",
        "TENSOR_SPLIT": "1,1",
        "MAIN_GPU": "0",
        "FIT": "true",
        "FIT_TARGET": "1536,1536",
        "FLASH_ATTN": "auto",
        "CACHE_TYPE_K": "q8_0",
        "CACHE_TYPE_V": "q8_0",
        "CONT_BATCHING": "true",
        "CACHE_PROMPT": "true",
        "JINJA": "true",
        "REASONING": "auto",
        "REASONING_FORMAT": "auto",
        "REASONING_EFFORT": "default",
        "REASONING_BUDGET": "-1",
        "METRICS": "true",
        "WEBUI": "true",
        "LOAD_MODE": "auto",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "EXTRA_ARGS": "",
    }

    def __init__(self, **values: str) -> None:
        data = {**self.DEFAULTS, **{k: v for k, v in values.items() if v is not None}}
        unknown = set(data) - set(self.DEFAULTS)
        if unknown:
            raise ConfigError(f"未知的 server 配置键：{sorted(unknown)}")
        for key, val in data.items():
            setattr(self, key, val)
        # 路径类键统一展开 ~（与 Bash 版 load_config 行为一致）
        self.MODEL_DIR = expand_home(self.MODEL_DIR)
        self.MODEL = expand_home(self.MODEL)
        self.MM_PROJ = expand_home(self.MM_PROJ)

    def as_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in SERVER_CONFIG_KEYS}

    def validate(self) -> list[str]:
        """返回全部校验错误；空列表表示通过。规则与 Bash 版一一对应。"""
        errors: list[str] = []

        def need(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        def re_full(pattern: str, value: str) -> bool:
            return re.fullmatch(pattern, value) is not None

        need(bool(self.MODEL_DIR) and self.MODEL_DIR.startswith("/"), "MODEL_DIR 必须是绝对路径。")
        need(self.MODEL == "" or self.MODEL.startswith("/"), "MODEL 必须留空或为绝对 GGUF 路径。")
        need(self.MM_PROJ == "" or self.MM_PROJ.startswith("/"), "MM_PROJ 必须留空或为绝对路径。")
        need(bool(self.HOST) and not any(c.isspace() for c in self.HOST), "HOST 无效。")

        port_ok = self.PORT.isdigit() and 1 <= int(self.PORT) <= 65535
        need(port_ok, "PORT 必须在 1..65535。")

        for var in ("CTX_SIZE", "N_PARALLEL", "BATCH_SIZE", "UBATCH_SIZE", "THREADS", "THREADS_BATCH"):
            value = getattr(self, var)
            need(re_full(r"[1-9][0-9]*", value), f"{var} 必须为正整数。")

        need(
            self.MAIN_GPU.isdigit(),
            "MAIN_GPU 必须为非负整数。",
        )
        need(
            self.N_GPU_LAYERS in ("all", "auto") or self.N_GPU_LAYERS.isdigit(),
            "N_GPU_LAYERS 必须为 all、auto 或非负整数。",
        )
        need(re_full(r"none|layer|row|tensor", self.SPLIT_MODE), "SPLIT_MODE 无效。")
        need(
            re_full(r"[0-9]+([.][0-9]+)?(,[0-9]+([.][0-9]+)?)+", self.TENSOR_SPLIT),
            "TENSOR_SPLIT 格式应类似 1,1。",
        )
        need(re_full(r"[0-9]+(,[0-9]+)*", self.FIT_TARGET), "FIT_TARGET 格式应类似 1536,1536。")
        need(re_full(r"on|off|auto", self.FLASH_ATTN), "FLASH_ATTN 无效。")
        cache_types = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
        need(self.CACHE_TYPE_K in cache_types, "CACHE_TYPE_K 无效。")
        need(self.CACHE_TYPE_V in cache_types, "CACHE_TYPE_V 无效。")
        need(re_full(r"on|off|auto", self.REASONING), "REASONING 无效。")
        need(
            re_full(r"auto|none|deepseek|deepseek-legacy", self.REASONING_FORMAT),
            "REASONING_FORMAT 无效。",
        )
        need(
            re_full(r"default|minimal|low|medium|high|xhigh|max", self.REASONING_EFFORT),
            "REASONING_EFFORT 无效。",
        )
        need(re_full(r"-1|[0-9]+", self.REASONING_BUDGET), "REASONING_BUDGET 必须为 -1 或非负整数。")
        need(
            re_full(r"auto|none|mmap|mlock|mmap\+mlock|dio", self.LOAD_MODE),
            "LOAD_MODE 无效。",
        )
        need(re_full(r"[0-9]+(,[0-9]+)*", self.CUDA_VISIBLE_DEVICES), "CUDA_VISIBLE_DEVICES 无效。")

        for var in ("FIT", "CONT_BATCHING", "CACHE_PROMPT", "JINJA", "METRICS", "WEBUI"):
            value = getattr(self, var)
            need(value in ("true", "false"), f"{var} 只能是 true 或 false。")

        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ConfigError("配置无效：" + "；".join(errors))


class BuildConfig:
    """编译参数；不参与服务启动。"""

    DEFAULTS: dict[str, str] = {
        "CUDA_ARCHITECTURES": "86;89",
        "BUILD_JOBS": "auto",
        "LLAMACPP_REF": "master",
    }

    def __init__(self, **values: str) -> None:
        data = {**self.DEFAULTS, **{k: v for k, v in values.items() if v is not None}}
        unknown = set(data) - set(self.DEFAULTS)
        if unknown:
            raise ConfigError(f"未知的 build 配置键：{sorted(unknown)}")
        for key, val in data.items():
            setattr(self, key, val)

    def validate(self) -> list[str]:
        errors: list[str] = []
        jobs = self.BUILD_JOBS
        if not (jobs == "auto" or re.fullmatch(r"[1-9][0-9]*", jobs)):
            errors.append("BUILD_JOBS 必须为 auto 或正整数。")
        if not re.fullmatch(r"[0-9]+([;][0-9]+)*", self.CUDA_ARCHITECTURES):
            errors.append("CUDA_ARCHITECTURES 格式应类似 86;89。")
        ref = self.LLAMACPP_REF
        if not ref or any(c.isspace() for c in ref):
            errors.append("LLAMACPP_REF 无效。")
        return errors


def parse_env_text(text: str, kind: str) -> tuple[dict[str, str], list[str]]:
    """解析 env 文本，返回 (键值字典, 告警列表)。kind 为 server/build/legacy。"""
    if kind == "server":
        allowed = SERVER_CONFIG_KEYS
    elif kind == "build":
        allowed = BUILD_CONFIG_KEYS
    elif kind == "legacy":
        allowed = CONFIG_KEYS
    else:
        raise ValueError(f"未知配置类型：{kind}")

    values: dict[str, str] = {}
    warnings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key == "LLAMA_CPP_REF" and kind != "server":
            warnings.append("配置键 LLAMA_CPP_REF 已更名为 LLAMACPP_REF；本次兼容读取旧值。")
            key = "LLAMACPP_REF"
        if key in allowed:
            values[key] = value
        else:
            warnings.append(f"忽略未知配置项：{key}")
    return values, warnings


def parse_env_file(path: Path, kind: str) -> tuple[dict[str, str], list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, []
    return parse_env_text(text, kind)


def render_server_config(cfg: ServerConfig) -> str:
    """渲染为与 Bash 版 write_server_config 完全相同的文本布局。"""
    d = cfg.as_dict()
    lines = [
        "# llama-server startup configuration. Edit values, then run:",
        "#   systemctl --user restart llamacpp.service",
        "# Parsed as data by llamacpp; never sourced as shell code.",
        "",
        "# Model",
        f"MODEL_DIR={d['MODEL_DIR']}",
        f"MODEL={d['MODEL']}",
        f"MODEL_ALIAS={d['MODEL_ALIAS']}",
        f"MM_PROJ={d['MM_PROJ']}",
        "",
        "# OpenAI-compatible HTTP server",
        f"HOST={d['HOST']}",
        f"PORT={d['PORT']}",
        f"API_KEY={d['API_KEY']}",
        "",
        "# Context, concurrency and CPU threads",
        f"CTX_SIZE={d['CTX_SIZE']}",
        f"N_PARALLEL={d['N_PARALLEL']}",
        f"BATCH_SIZE={d['BATCH_SIZE']}",
        f"UBATCH_SIZE={d['UBATCH_SIZE']}",
        f"THREADS={d['THREADS']}",
        f"THREADS_BATCH={d['THREADS_BATCH']}",
        "",
        "# Dual-GPU offload and memory",
        f"N_GPU_LAYERS={d['N_GPU_LAYERS']}",
        f"SPLIT_MODE={d['SPLIT_MODE']}",
        f"TENSOR_SPLIT={d['TENSOR_SPLIT']}",
        f"MAIN_GPU={d['MAIN_GPU']}",
        f"FIT={d['FIT']}",
        f"FIT_TARGET={d['FIT_TARGET']}",
        f"FLASH_ATTN={d['FLASH_ATTN']}",
        f"CACHE_TYPE_K={d['CACHE_TYPE_K']}",
        f"CACHE_TYPE_V={d['CACHE_TYPE_V']}",
        f"CUDA_VISIBLE_DEVICES={d['CUDA_VISIBLE_DEVICES']}",
        "",
        "# Chat template, reasoning and server features",
        f"CONT_BATCHING={d['CONT_BATCHING']}",
        f"CACHE_PROMPT={d['CACHE_PROMPT']}",
        f"JINJA={d['JINJA']}",
        f"REASONING={d['REASONING']}",
        f"REASONING_FORMAT={d['REASONING_FORMAT']}",
        f"REASONING_EFFORT={d['REASONING_EFFORT']}",
        f"REASONING_BUDGET={d['REASONING_BUDGET']}",
        f"METRICS={d['METRICS']}",
        f"WEBUI={d['WEBUI']}",
        f"LOAD_MODE={d['LOAD_MODE']}",
        "",
        "# Additional native llama-server options. Avoid duplicating managed options above.",
        f"EXTRA_ARGS={d['EXTRA_ARGS']}",
    ]
    return "\n".join(lines) + "\n"


def render_build_config(cfg: BuildConfig) -> str:
    lines = [
        "# llamacpp build configuration; parsed as data and never sourced.",
        f"CUDA_ARCHITECTURES={cfg.CUDA_ARCHITECTURES}",
        f"BUILD_JOBS={cfg.BUILD_JOBS}",
        f"LLAMACPP_REF={cfg.LLAMACPP_REF}",
    ]
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    """原子写文件并设置权限，避免写入中途损坏配置。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_server_config(cfg: ServerConfig, path: Path, dry_run: bool = False) -> None:
    cfg.require_valid()
    for key, value in cfg.as_dict().items():
        if "\n" in value or "\r" in value:
            raise ConfigError(f"配置 {key} 不能包含换行。")
    if dry_run:
        print(f"[DRY-RUN] 将写入 Server 启动配置：{path}")
        return
    atomic_write(path, render_server_config(cfg))


def save_build_config(cfg: BuildConfig, path: Path, dry_run: bool = False) -> None:
    errors = cfg.validate()
    if errors:
        raise ConfigError("配置无效：" + "；".join(errors))
    if dry_run:
        print(f"[DRY-RUN] 将写入 build 配置：{path}")
        return
    atomic_write(path, render_build_config(cfg))


def load_server_config(path: Path) -> tuple[ServerConfig, list[str]]:
    values, warnings = parse_env_file(path, "server")
    return ServerConfig(**values), warnings


def load_build_config(path: Path) -> tuple[BuildConfig, list[str]]:
    values, warnings = parse_env_file(path, "build")
    return BuildConfig(**values), warnings


def ensure_configs(
    server_path: Path,
    build_path: Path,
    legacy_path: Path | None = None,
    dry_run: bool = False,
) -> tuple[ServerConfig, BuildConfig]:
    """确保两个配置文件存在；支持从 legacy config.env 单次迁移。"""
    if (
        not server_path.exists()
        and not build_path.exists()
        and legacy_path is not None
        and legacy_path.exists()
    ):
        values, _ = parse_env_file(legacy_path, "legacy")
        server_values = {k: v for k, v in values.items() if k in SERVER_CONFIG_KEYS}
        build_values = {k: v for k, v in values.items() if k in BUILD_CONFIG_KEYS}
        server_cfg = ServerConfig(**server_values)
        build_cfg = BuildConfig(**build_values)
        save_server_config(server_cfg, server_path, dry_run)
        save_build_config(build_cfg, build_path, dry_run)
        return server_cfg, build_cfg

    server_cfg, _ = load_server_config(server_path)
    build_cfg, _ = load_build_config(build_path)
    if not server_path.exists():
        save_server_config(server_cfg, server_path, dry_run)
    if not build_path.exists():
        save_build_config(build_cfg, build_path, dry_run)
    server_cfg.require_valid()
    return server_cfg, build_cfg
