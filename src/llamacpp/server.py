"""llama-server / llama-bench 命令构建与健康检查，与 Bash 版参数一一对应。"""

from __future__ import annotations

import shlex
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ServerConfig


def parse_extra_args(extra: str) -> list[str]:
    """按 POSIX 规则拆分 EXTRA_ARGS（与 Bash 版的 shlex 行为一致）。"""
    return shlex.split(extra) if extra.strip() else []


def build_server_command(cfg: ServerConfig, server_bin: Path) -> list[str]:
    argv: list[str] = [
        str(server_bin),
        "--model", cfg.MODEL,
        "--host", cfg.HOST,
        "--port", cfg.PORT,
        "--ctx-size", cfg.CTX_SIZE,
        "--parallel", cfg.N_PARALLEL,
        "--batch-size", cfg.BATCH_SIZE,
        "--ubatch-size", cfg.UBATCH_SIZE,
        "--threads", cfg.THREADS,
        "--threads-batch", cfg.THREADS_BATCH,
        "--n-gpu-layers", cfg.N_GPU_LAYERS,
        "--split-mode", cfg.SPLIT_MODE,
        "--tensor-split", cfg.TENSOR_SPLIT,
        "--main-gpu", cfg.MAIN_GPU,
        "--flash-attn", cfg.FLASH_ATTN,
        "--cache-type-k", cfg.CACHE_TYPE_K,
        "--cache-type-v", cfg.CACHE_TYPE_V,
        "--load-mode", cfg.LOAD_MODE,
        "--reasoning", cfg.REASONING,
        "--log-timestamps",
    ]
    if cfg.MODEL_ALIAS:
        argv += ["--alias", cfg.MODEL_ALIAS]
    if cfg.MM_PROJ:
        argv += ["--mmproj", cfg.MM_PROJ]
    if cfg.API_KEY:
        argv += ["--api-key", cfg.API_KEY]
    if cfg.FIT == "true":
        argv += ["--fit", "on", "--fit-target", cfg.FIT_TARGET]
    else:
        argv += ["--fit", "off"]
    argv.append("--cont-batching" if cfg.CONT_BATCHING == "true" else "--no-cont-batching")
    argv.append("--cache-prompt" if cfg.CACHE_PROMPT == "true" else "--no-cache-prompt")
    argv.append("--jinja" if cfg.JINJA == "true" else "--no-jinja")
    argv.append("--webui" if cfg.WEBUI == "true" else "--no-webui")
    if cfg.METRICS == "true":
        argv.append("--metrics")
    if cfg.REASONING_FORMAT != "auto":
        argv += ["--reasoning-format", cfg.REASONING_FORMAT]
    if cfg.REASONING_EFFORT != "default":
        argv += ["--reasoning-effort", cfg.REASONING_EFFORT]
    if cfg.REASONING_BUDGET != "-1":
        argv += ["--reasoning-budget", cfg.REASONING_BUDGET]
    argv.extend(parse_extra_args(cfg.EXTRA_ARGS))
    return argv


def build_bench_command(cfg: ServerConfig, bench_bin: Path) -> list[str]:
    bench_ngl = "-1" if cfg.N_GPU_LAYERS in ("all", "auto") else cfg.N_GPU_LAYERS
    tensor_split = cfg.TENSOR_SPLIT.replace(",", "/")
    return [
        str(bench_bin),
        "-m", cfg.MODEL,
        "-ngl", bench_ngl,
        "-sm", cfg.SPLIT_MODE,
        "-ts", tensor_split,
        "-mg", cfg.MAIN_GPU,
        "-t", cfg.THREADS,
        "-b", cfg.BATCH_SIZE,
        "-ub", cfg.UBATCH_SIZE,
        "-ctk", cfg.CACHE_TYPE_K,
        "-ctv", cfg.CACHE_TYPE_V,
        "-fa", cfg.FLASH_ATTN,
        "-lm", cfg.LOAD_MODE,
        "--progress",
    ]


def normalize_host(host: str) -> str:
    """把监听地址转换为可连接的地址（通配地址映射到回环）。"""
    if host in ("0.0.0.0", ""):
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "[::1]"
    if host.startswith("["):
        return host
    if ":" in host:
        return f"[{host}]"
    return host


def health_url(host: str, port: str | int) -> str:
    """把监听地址转换为可访问的 health URL（与 Bash 版映射规则一致）。"""
    return f"http://{normalize_host(host)}:{port}/health"


def http_get(url: str, api_key: str | None = None, timeout: float = 10.0) -> tuple[int, bytes]:
    """返回 (HTTP 状态码, 响应体)；网络错误抛出 urllib.error.URLError。"""
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def wait_for_health(
    url: str,
    is_alive: callable,
    timeout_seconds: int = 180,
    on_progress=None,
) -> bool:
    """轮询 health 端点；is_alive 为 False 时立即放弃。"""
    elapsed = 0
    while elapsed < timeout_seconds:
        if not is_alive():
            return False
        try:
            code, _ = http_get(url, timeout=3)
        except (urllib.error.URLError, socket.error):
            code = 0
        if code == 200:
            return True
        if on_progress is not None and elapsed % 15 == 0:
            on_progress(code, elapsed)
        time.sleep(3)
        elapsed += 3
    return False
