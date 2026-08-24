"""llamacpp 命令行入口（typer）。

子命令与 Bash 版一一对应：install / update / self-install / start / stop /
restart / status / logs / config / models / doctor / self-test / bench /
driver / uninstall / menu / run-server。
"""

from __future__ import annotations

import dataclasses
import getpass
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import typer

from . import __version__, paths, service as svc
from .config import (
    BUILD_CONFIG_KEYS,
    SERVER_CONFIG_KEYS,
    BuildConfig,
    ConfigError,
    ServerConfig,
    atomic_write,
    ensure_configs,
    load_build_config,
    load_server_config,
    parse_env_file,
    save_build_config,
    save_server_config,
)
from .gpu import detect_gpus, gpu_models_are_mixed, list_gpus, low_memory_count, nvidia_smi_available
from .installer import (
    CommandError,
    assert_arch_linux,
    build_llama_cpp,
    die,
    git_short_commit,
    info,
    install_build_dependencies,
    manage_driver,
    prepare_source,
    safe_remove_tree,
    sh,
    source_tree_is_clean,
    warn,
)
from .models import choose_model, scan_models
from .server import build_bench_command, build_server_command, health_url, http_get, wait_for_health

app = typer.Typer(
    name="llamacpp",
    help="Arch Linux 双 NVIDIA GPU llama.cpp 部署与管理器（Python 版）",
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)

EXPECTED_GPU_COUNT = 2
MIN_GPU_MEMORY_MIB = 11000
CONFLICTING_SERVICES = ("vllm-dual-3060.service", "llama-cpp-dual-gpu.service")


@dataclasses.dataclass
class State:
    dry_run: bool = False
    non_interactive: bool = False
    verbose: bool = False


STATE = State()


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and not STATE.non_interactive


def _confirm(prompt: str) -> bool:
    if not _interactive():
        return False
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _load_all() -> tuple[ServerConfig, BuildConfig]:
    try:
        return ensure_configs(
            paths.server_config_file(), paths.build_config_file(), paths.legacy_config_file()
        )
    except ConfigError as exc:
        die(str(exc))


def _check_gpu_or_die() -> None:
    if not detect_gpus(EXPECTED_GPU_COUNT, MIN_GPU_MEMORY_MIB):
        for gpu in list_gpus():
            print(gpu)
        die(f"需要 {EXPECTED_GPU_COUNT} 张可用 NVIDIA GPU，且每张显存至少约 "
            f"{MIN_GPU_MEMORY_MIB // 1024} GiB。")
    if gpu_models_are_mixed():
        warn("检测到不同型号 GPU；layer split 可用，但速度受较慢的卡限制。")


def _check_conflicting_services() -> None:
    for name in CONFLICTING_SERVICES:
        if svc.is_active(name):
            die(f"{name} 正在占用 GPU；请先停止该服务。")


# ---------------------------------------------------------------- install --

@app.command()
def install(
    ref: str = typer.Option("master", "--ref", help="Git 分支、tag 或 commit"),
    driver: str = typer.Option("none", "--driver", help="none、nvidia-open 或 nvidia-open-dkms"),
    start: bool = typer.Option(False, "--start", help="安装后启动服务"),
    enable_linger: bool = typer.Option(False, "--enable-linger", help="退出登录后继续运行（需 sudo）"),
) -> None:
    """安装构建依赖、编译 llama.cpp 并生成 systemd user service。"""
    if driver not in ("none", "nvidia-open", "nvidia-open-dkms"):
        die("driver 选项无效。")
    assert_arch_linux()
    install_build_dependencies(non_interactive=STATE.non_interactive, dry_run=STATE.dry_run)
    if driver != "none":
        manage_driver(driver, non_interactive=STATE.non_interactive, dry_run=STATE.dry_run)
    _, build_cfg = _load_all()
    build_cfg.LLAMACPP_REF = ref
    save_build_config(build_cfg, paths.build_config_file())
    prepare_source(paths.source_dir(), ref, dry_run=STATE.dry_run)
    build_llama_cpp(
        paths.source_dir(), paths.build_dir(),
        build_cfg.CUDA_ARCHITECTURES, build_cfg.BUILD_JOBS, dry_run=STATE.dry_run,
    )
    _write_launcher_and_unit(dry_run=STATE.dry_run)
    model_dir = Path(load_server_config(paths.server_config_file())[0].MODEL_DIR)
    if not STATE.dry_run:
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[DRY-RUN] mkdir -p {model_dir}")
    if enable_linger:
        sh(["sudo", "loginctl", "enable-linger", os.environ.get("USER", "")], dry_run=STATE.dry_run)
    info(f"安装完成。Server 启动配置：{paths.server_config_file()}")
    if driver != "none":
        warn("驱动已变更，请重启后再启动。")
    elif start:
        start_service()
    else:
        info("下一步：llamacpp doctor && llamacpp start")


@app.command()
def update(
    ref: str | None = typer.Option(None, "--ref", help="覆盖配置文件中的 ref"),
    start: bool = typer.Option(False, "--start", help="更新后启动服务"),
) -> None:
    """拉取并重新编译最新 llama.cpp，保留配置。"""
    _, build_cfg = _load_all()
    target_ref = ref or build_cfg.LLAMACPP_REF
    was_active = svc.is_active(__package_service_name())
    if was_active:
        stop()
    install_build_dependencies(non_interactive=STATE.non_interactive, dry_run=STATE.dry_run)
    build_cfg.LLAMACPP_REF = target_ref
    save_build_config(build_cfg, paths.build_config_file())
    prepare_source(paths.source_dir(), target_ref, dry_run=STATE.dry_run)
    build_llama_cpp(
        paths.source_dir(), paths.build_dir(),
        build_cfg.CUDA_ARCHITECTURES, build_cfg.BUILD_JOBS, dry_run=STATE.dry_run,
    )
    _write_launcher_and_unit(dry_run=STATE.dry_run)
    if was_active or start:
        start_service()
    commit = git_short_commit(paths.source_dir())
    info(f"更新完成：{commit or 'dry-run'}")


@app.command("self-install")
def self_install() -> None:
    """刷新启动器与 user service 文件。"""
    _load_all()
    _write_launcher_and_unit(dry_run=STATE.dry_run)
    info(f"启动器与 service 已刷新；启动配置:{paths.server_config_file()}")


def __package_service_name() -> str:
    from . import SERVICE_NAME

    return SERVICE_NAME


def _current_manager_path() -> str:
    candidate = Path(sys.argv[0]).resolve()
    return str(candidate)


def _write_launcher_and_unit(dry_run: bool = False) -> None:
    from .service import render_launcher, render_unit

    launcher_path = paths.launcher_bin()
    service_path = paths.service_file()
    manager_path = _current_manager_path()

    if dry_run:
        print(f"[DRY-RUN] 将写入启动器：{launcher_path}")
        print(f"[DRY-RUN] 将写入 user service：{service_path}")
        svc.daemon_reload(dry_run=True)
        return
    from .service import render_unit

    # 启动器固定指向当前 CLI 可执行文件，保证 run-server 走同一套 Python 逻辑
    manager_path = _current_manager_path()
    launcher_text = (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f'exec "{manager_path}" run-server "$@"\n'
    )
    atomic_write(launcher_path, launcher_text, mode=0o755)
    atomic_write(service_path, render_unit(launcher_name=launcher_path.name), mode=0o644)
    svc.daemon_reload()
    svc.enable(service_path.name)


# ------------------------------------------------------------ 生命周期 ----


@app.command()
def start() -> None:
    """启动 llama-server 服务并等待就绪。"""
    start_service()


def start_service() -> None:
    cfg, _ = _load_all()
    if not cfg.MODEL and _interactive():
        models_command(write_back=True)
        cfg, _ = _load_all()
    if not cfg.MODEL:
        die("未选择模型；请先运行 models。")
    _check_gpu_or_die()
    _check_conflicting_services()
    service_path = paths.service_file()
    if not svc.service_file_exists(service_path):
        die("user service 未安装；请先运行 install。")
    name = service_path.name
    svc.start(name, dry_run=STATE.dry_run)
    if STATE.dry_run:
        return
    time.sleep(2)
    if not svc.is_active(name):
        print(svc.status_text(name))
        die("服务启动失败；运行 logs 查看错误。")
    url = health_url(cfg.HOST, cfg.PORT)

    def progress(code: int, elapsed: int) -> None:
        info(f"模型仍在加载（health HTTP {code}，已等待 {elapsed}s）……")

    if wait_for_health(url, lambda: svc.is_active(name), on_progress=progress):
        info(f"服务就绪：OpenAI http://127.0.0.1:{cfg.PORT}/v1；Web UI http://127.0.0.1:{cfg.PORT}/")
    else:
        warn("服务仍在运行但 180 秒内未就绪；请运行 logs -f。")


@app.command()
def stop() -> None:
    """停止服务。"""
    service_path = paths.service_file()
    if not svc.service_file_exists(service_path):
        warn("service 未安装。")
        return
    name = service_path.name
    if svc.is_active(name):
        svc.stop(name, dry_run=STATE.dry_run)
        info("服务已停止。")
    else:
        info("服务已经停止。")


@app.command()
def restart() -> None:
    """重启服务。"""
    stop()
    start_service()


@app.command()
def status() -> None:
    """查看服务、GPU、构建与配置摘要。"""
    cfg, build_cfg = _load_all()
    print("\n=== GPU ===")
    if nvidia_smi_available():
        for gpu in list_gpus():
            print(f"{gpu.index}, {gpu.name}, {gpu.memory_total_mib} MiB, "
                  f"free {gpu.memory_free_mib} MiB, driver {gpu.driver_version}")
    else:
        print("nvidia-smi 不可用")
    print("\n=== 服务 ===")
    service_path = paths.service_file()
    if svc.service_file_exists(service_path):
        print(svc.status_text(service_path.name))
    else:
        print(f"service 未安装：{service_path}")
    print("\n=== 构建 ===")
    server_bin = paths.server_bin()
    if server_bin.is_file() and os.access(server_bin, os.X_OK):
        result = subprocess.run([str(server_bin), "--version"], capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())
    else:
        print("llama-server 未编译")
    from .installer import git_short_commit

    commit = git_short_commit(paths.source_dir())
    if commit:
        print(f"Git commit: {commit}")
    print("\n=== 配置 ===")
    show_config_values(cfg, build_cfg)
    print(f"\nOpenAI API: http://{cfg.HOST}:{cfg.PORT}/v1")
    print(f"Web UI:     http://{cfg.HOST}:{cfg.PORT}/")


@app.command()
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="持续跟踪"),
    lines: int = typer.Option(100, "--lines", "-n", min=1, help="行数"),
) -> None:
    """查看 journalctl 日志。"""
    service_path = paths.service_file()
    args = ["journalctl", "--user", "-u", service_path.name, "-n", str(lines), "--no-pager"]
    if follow:
        args.append("-f")
    raise typer.Exit(subprocess.run(args, check=False).returncode)


# ---------------------------------------------------------------- 配置 ----


@app.command()
def config(
    key: str = typer.Argument(None, help="配置键"),
    value: str = typer.Argument(None, help="配置值"),
    show: bool = typer.Option(False, "--show", help="显示当前配置"),
    edit: bool = typer.Option(False, "--edit", help="编辑 server.env"),
    edit_build: bool = typer.Option(False, "--edit-build", help="编辑 build.env"),
    wizard: bool = typer.Option(False, "--wizard", help="交互式配置向导"),
) -> None:
    """管理 server.env / build.env。无参数时进入向导（交互终端）。"""
    cfg, build_cfg = _load_all()
    if show:
        show_config_values(cfg, build_cfg)
        return
    if edit or edit_build:
        _edit_config(paths.build_config_file() if edit_build else paths.server_config_file())
        return
    if wizard or (key is None and value is None):
        _config_wizard(cfg)
        return
    if key is None or value is None:
        die("用法：config KEY VALUE；或 config --show / --edit / --edit-build / --wizard")
    all_keys = SERVER_CONFIG_KEYS + BUILD_CONFIG_KEYS
    if key not in all_keys:
        die(f"不支持的配置键：{key}")
    if "\n" in value or "\r" in value:
        die("配置值不能包含换行。")
    if key in BUILD_CONFIG_KEYS:
        setattr(build_cfg, key, value)
        errors = build_cfg.validate()
        if errors:
            die("；".join(errors))
        save_build_config(build_cfg, paths.build_config_file())
    else:
        if key in ("MODEL_DIR", "MODEL", "MM_PROJ"):
            from .config import expand_home

            value = expand_home(value)
        setattr(cfg, key, value)
        errors = cfg.validate()
        if errors:
            die("；".join(errors))
        save_server_config(cfg, paths.server_config_file())
    info(f"已更新 {key}。")


def show_config_values(cfg: ServerConfig, build_cfg: BuildConfig) -> None:
    d = cfg.as_dict()

    def redact(value: str) -> str:
        return "********" if value else "(未设置)"

    print("=== Server 启动配置 ===")
    print(f"文件: {paths.server_config_file()}")
    for key in SERVER_CONFIG_KEYS:
        display = d[key]
        if key == "MODEL" and not display:
            display = "(未选择)"
        elif key == "MODEL_ALIAS" and not display:
            display = "(自动)"
        elif key == "MM_PROJ" and not display:
            display = "(未设置)"
        elif key == "API_KEY":
            display = redact(display)
        print(f"{key}={display}")
    print("\n=== 编译配置（不参与服务启动）===")
    print(f"文件: {paths.build_config_file()}")
    print(f"CUDA_ARCHITECTURES={build_cfg.CUDA_ARCHITECTURES}")
    print(f"BUILD_JOBS={build_cfg.BUILD_JOBS}")
    print(f"LLAMACPP_REF={build_cfg.LLAMACPP_REF}")


def _prompt_value(label: str, current: str, secret: bool = False) -> str:
    hint = "(已设置；直接回车保留)" if secret and current else f"[{current}]"
    if secret:
        answer = getpass.getpass(f"{label} {hint}: ")
    else:
        answer = input(f"{label} {hint}: ")
    return answer or current


def _config_wizard(cfg: ServerConfig) -> None:
    if not _interactive():
        die("交互配置需要终端；自动化请使用 config KEY VALUE。")
    print("\n直接回车保留当前值。布尔值使用 true/false。\n")
    cfg.MODEL_DIR = _prompt_value("模型目录", cfg.MODEL_DIR)
    from .config import expand_home

    cfg.MODEL_DIR = expand_home(cfg.MODEL_DIR)
    if _confirm("现在扫描并选择 GGUF 模型吗？"):
        models_command(write_back=True)
        cfg, _ = _load_all()
    fields = [
        ("MODEL_ALIAS", "API 模型别名（空=文件名）", False),
        ("MM_PROJ", "多模态 mmproj 路径（空=无）", False),
        ("HOST", "监听地址", False),
        ("PORT", "端口", False),
        ("CTX_SIZE", "上下文长度", False),
        ("N_PARALLEL", "并发 slots", False),
        ("BATCH_SIZE", "逻辑 batch size", False),
        ("UBATCH_SIZE", "物理 ubatch size", False),
        ("THREADS", "生成/解码 CPU 线程数", False),
        ("THREADS_BATCH", "提示词处理 CPU 线程数", False),
        ("SPLIT_MODE", "双卡 split mode: layer/row/tensor/none", False),
        ("TENSOR_SPLIT", "双卡比例", False),
        ("FIT_TARGET", "每卡预留显存 MiB", False),
        ("FLASH_ATTN", "Flash Attention: auto/on/off", False),
        ("CACHE_TYPE_K", "K cache 类型", False),
        ("CACHE_TYPE_V", "V cache 类型", False),
        ("REASONING", "Reasoning: auto/on/off", False),
        ("REASONING_EFFORT", "Reasoning effort", False),
        ("REASONING_BUDGET", "Reasoning budget，-1=不限", False),
        ("EXTRA_ARGS", "额外 llama-server 参数", False),
    ]
    for key, label, secret in fields:
        setattr(cfg, key, _prompt_value(label, getattr(cfg, key), secret))
    cfg.API_KEY = _prompt_value("API Key（直接回车保留）", cfg.API_KEY, secret=True)
    cfg.MM_PROJ = expand_home(cfg.MM_PROJ)
    errors = cfg.validate()
    if errors:
        die("；".join(errors))
    save_server_config(cfg, paths.server_config_file())
    info("配置完成。")


def _edit_config(file: Path) -> None:
    editor = os.environ.get("EDITOR") or ("/usr/bin/nano" if Path("/usr/bin/nano").exists() else "vi")
    backup = file.with_suffix(f".backup.{os.getpid()}")
    backup.write_text(file.read_text(encoding="utf-8"))
    subprocess.run([editor, str(file)], check=False)
    kind = "build" if file.name == "build.env" else "server"
    try:
        values, warnings = parse_env_file(file, kind)
        if kind == "server":
            parsed: ServerConfig | BuildConfig = ServerConfig(**values)
        else:
            parsed = BuildConfig(**values)
        errors = parsed.validate()
        if errors:
            raise ConfigError("；".join(errors))
    except ConfigError as exc:
        backup.replace(file)
        die(f"配置无效，已恢复编辑前版本：{exc}")
    finally:
        backup.unlink(missing_ok=True)
    for warning in warnings or []:
        warn(warning)
    info(f"配置有效并已保存：{file}")


# ---------------------------------------------------------------- 模型 ----


@app.command()
def models(write_back: bool = typer.Option(False, "--write-back", hidden=True)) -> None:
    """扫描 ~/models 下的 GGUF 并选择。"""
    models_command(write_back=write_back)


def models_command(write_back: bool = True) -> None:
    cfg, _ = _load_all()
    root = Path(cfg.MODEL_DIR)
    found = scan_models(root)
    if not found:
        warn(f"在 {root} 下没有找到可用 GGUF。")
        raise typer.Exit(1)
    if not _interactive():
        if len(found) == 1:
            cfg.MODEL = str(found[0])
            if not cfg.MODEL_ALIAS:
                cfg.MODEL_ALIAS = found[0].stem
            save_server_config(cfg, paths.server_config_file())
            info(f"自动选择唯一模型：{cfg.MODEL}")
            return
        warn("发现多个模型；非交互模式不会擅自选择。")
        raise typer.Exit(1)
    chosen = choose_model(found)
    if chosen is None:
        raise typer.Exit(1)
    cfg.MODEL = str(chosen)
    if not cfg.MODEL_ALIAS:
        cfg.MODEL_ALIAS = chosen.stem
    save_server_config(cfg, paths.server_config_file())
    info(f"已选择：{cfg.MODEL}")


# ------------------------------------------------------- 自检与基准 --------


@app.command()
def doctor() -> None:
    """检查系统、驱动、CUDA、编译结果与配置。"""
    failures = 0
    warnings = 0

    def ok(msg: str) -> None:
        nonlocal failures
        print(f"[OK] {msg}")

    def fail(msg: str) -> None:
        nonlocal failures
        failures += 1
        print(f"[FAIL] {msg}")

    def soft_warn(msg: str) -> None:
        nonlocal warnings
        warnings += 1
        print(f"[WARN] {msg}")

    def note(msg: str) -> None:
        print(f"[INFO] {msg}")

    print(f"=== llamacpp doctor（Python v{__version__}） ===")
    if Path("/etc/arch-release").exists():
        ok("Arch Linux")
    else:
        fail("不是 Arch Linux")

    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    note(f"CPU: {cpu_model or '未知'}；逻辑核心 {os.cpu_count()}")
    meminfo = Path("/proc/meminfo")
    ram_kib = 0
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_kib = int(line.split()[1])
                break
    if ram_kib >= 60_000_000:
        ok(f"RAM: 约 {ram_kib // 1024 // 1024} GiB")
    else:
        soft_warn(f"RAM: {ram_kib // 1024 // 1024} GiB")

    if nvidia_smi_available():
        gpus = list_gpus()
        for gpu in gpus:
            print(f"  GPU{gpu.index}: {gpu.name} {gpu.memory_total_mib} MiB "
                  f"(free {gpu.memory_free_mib})")
        if len(gpus) == EXPECTED_GPU_COUNT:
            ok("GPU 数量: 2")
        else:
            fail(f"GPU 数量: {len(gpus)}")
        low = sum(1 for g in gpus if g.memory_total_mib < MIN_GPU_MEMORY_MIB)
        if low == 0:
            ok(f"每张 GPU 显存不少于约 {MIN_GPU_MEMORY_MIB // 1024} GiB")
        else:
            fail(f"{low} 张 GPU 显存不足")
        names = {g.name for g in gpus}
        if len(names) > 1:
            soft_warn("混合 GPU：性能受较慢卡限制")
    else:
        fail("nvidia-smi 不可用")

    from .installer import installed_driver_package

    note(f"驱动包: {installed_driver_package() or '未安装'}")
    if Path("/opt/cuda/bin/nvcc").is_file():
        result = subprocess.run(["/opt/cuda/bin/nvcc", "--version"], capture_output=True, text=True)
        last_line = (result.stdout or "").strip().splitlines()[-1:] or ["未知版本"]
        ok(last_line[0])
    else:
        fail("CUDA Toolkit/nvcc 未安装")
    if Path("/usr/bin/cmake").exists() or any(Path(p).exists() for p in ["/usr/bin/cmake"]):
        ok("cmake 可用")
    else:
        fail("cmake 不可用")

    server_bin = paths.server_bin()
    if server_bin.is_file() and os.access(server_bin, os.X_OK):
        result = subprocess.run([str(server_bin), "--version"], capture_output=True, text=True)
        first = (result.stdout or "").splitlines()[:1]
        ok(first[0] if first else "llama-server 存在")
        ldd = subprocess.run(["ldd", str(server_bin)], capture_output=True, text=True)
        if "not found" in ldd.stdout:
            fail("llama-server 存在缺失动态库")
        else:
            ok("llama-server 动态库完整")
    else:
        fail(f"llama-server 未编译：{server_bin}")

    if paths.server_config_file().exists():
        cfg, build_cfg = _load_all()
        errors = cfg.validate() + build_cfg.validate()
        if errors:
            fail("配置无效：" + "；".join(errors))
        else:
            ok(f"配置有效：{paths.server_config_file()}")
    else:
        soft_warn("配置尚未生成")

    print(f"\n结果：{failures} 个失败，{warnings} 个警告。")
    if failures:
        raise typer.Exit(1)


@app.command("self-test")
def self_test() -> None:
    """HTTP health 与 OpenAI /v1/models 自检。"""
    cfg, _ = _load_all()
    name = paths.service_file().name
    if not svc.is_active(name):
        die("服务未运行。")
    url = health_url(cfg.HOST, cfg.PORT)
    try:
        code, body = http_get(url, cfg.API_KEY or None)
    except Exception as exc:  # noqa: BLE001
        die(f"health 请求失败：{exc}")
    if code != 200:
        die(f"health 返回 HTTP {code}")
    print(body.decode(errors="replace"))
    try:
        code, body = http_get(f"http://127.0.0.1:{cfg.PORT}/v1/models", cfg.API_KEY or None)
    except Exception as exc:  # noqa: BLE001
        die(f"/v1/models 请求失败：{exc}")
    if code != 200:
        die(f"/v1/models 返回 HTTP {code}")
    print(body.decode(errors="replace"))
    info("HTTP 自检通过。")


@app.command()
def bench() -> None:
    """使用当前模型执行 llama-bench（需先停止服务）。"""
    cfg, _ = _load_all()
    bench_bin = paths.bench_bin()
    if not bench_bin.is_file():
        die("llama-bench 未编译。")
    if not Path(cfg.MODEL).is_file():
        die(f"未选择有效模型：{cfg.MODEL}")
    if svc.is_active(paths.service_file().name):
        die("请先停止 llama.cpp service，避免显存冲突。")
    _check_conflicting_services()
    _check_gpu_or_die()
    argv = build_bench_command(cfg, bench_bin)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/opt/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    print("[BENCH COMMAND]", end=" ")
    for arg in argv:
        print(shlex.quote(arg), end=" ")
    print()
    result = subprocess.run(argv, env=env, check=False)
    raise typer.Exit(result.returncode)


@app.command()
def driver(
    dtype: str = typer.Option("nvidia-open-dkms", "--type", help="nvidia-open 或 nvidia-open-dkms"),
    repair: bool = typer.Option(False, "--repair", help="强制重装驱动包"),
) -> None:
    """可选安装/修复 NVIDIA open 驱动。"""
    manage_driver(dtype, repair=repair,
                  non_interactive=STATE.non_interactive, dry_run=STATE.dry_run)


@app.command()
def uninstall(
    purge: bool = typer.Option(False, "--purge", help="同时删除源码、构建目录和配置"),
    yes: bool = typer.Option(False, "--yes", help="跳过 purge 确认"),
) -> None:
    """移除程序与 service；默认保留模型与配置。"""
    service_path = paths.service_file()
    name = service_path.name
    if svc.service_file_exists(service_path):
        svc.stop(name, dry_run=STATE.dry_run)
        svc.disable(name, dry_run=STATE.dry_run)
    for target in (service_path, paths.launcher_bin()):
        if STATE.dry_run:
            print(f"[DRY-RUN] rm -f {target}")
        else:
            target.unlink(missing_ok=True)
    svc.daemon_reload(dry_run=STATE.dry_run)
    if purge:
        if not yes and not _confirm(
            f"将删除 {paths.install_root()} 和 {paths.config_dir()}，确定吗？"
        ):
            warn("已取消 purge；源码、构建和配置保留。")
            return
        safe_remove_tree(paths.install_root(), dry_run=STATE.dry_run)
        safe_remove_tree(paths.config_dir(), dry_run=STATE.dry_run)
        info("已删除源码、构建和配置。")
    else:
        info("已移除命令与 service；保留源码、构建、配置和模型。")
    info("模型目录永远不会被卸载命令删除。")


# ------------------------------------------------------------ 直启/菜单 ---


@app.command("run-server")
def run_server() -> None:
    """内部命令：以前台方式运行 llama-server。"""
    cfg, _ = _load_all()
    server_bin = paths.server_bin()
    if not (server_bin.is_file() and os.access(server_bin, os.X_OK)):
        die("llama-server 未编译；请先运行 install。")
    if not cfg.MODEL:
        die("尚未选择模型；请运行 models。")
    model = Path(cfg.MODEL)
    if not model.is_file():
        die(f"模型不存在:{model}")
    if model.suffix.lower() != ".gguf":
        die("llama.cpp 模型必须为 GGUF 文件。")
    if cfg.MM_PROJ and not Path(cfg.MM_PROJ).is_file():
        die(f"MM_PROJ 不存在:{cfg.MM_PROJ}")
    _check_gpu_or_die()
    _check_conflicting_services()
    if not cfg.API_KEY and cfg.HOST not in ("127.0.0.1", "localhost"):
        warn(f"服务监听 {cfg.HOST} 但未配置 API_KEY；局域网内任何设备都可访问。")
    argv = build_server_command(cfg, server_bin)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/opt/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    info(f"启动 llama-server：model={cfg.MODEL}, ctx={cfg.CTX_SIZE}, "
         f"split={cfg.SPLIT_MODE}/{cfg.TENSOR_SPLIT}")
    print("[COMMAND]", end=" ")
    for arg in argv:
        if cfg.API_KEY and arg == cfg.API_KEY:
            print(shlex.quote("********"), end=" ")
        else:
            print(shlex.quote(arg), end=" ")
    print()
    os.execve(argv[0], argv, env)


@app.command()
def menu() -> None:
    """交互式管理菜单。"""
    if not _interactive():
        die("menu 需要交互式终端。")
    actions = {
        "1": ("安装/编译 llama.cpp", lambda: install()),
        "2": ("扫描并选择 GGUF", lambda: models_command()),
        "3": ("配置向导", lambda: _wizard_entry()),
        "4": ("启动", start_service),
        "5": ("停止", stop),
        "6": ("重启", restart),
        "7": ("状态", status),
        "8": ("日志", logs),
        "9": ("Doctor", doctor),
        "10": ("HTTP 自检", self_test),
        "11": ("Benchmark", bench),
        "12": ("更新 llama.cpp", lambda: update()),
        "13": ("安装/修复 NVIDIA open 驱动", lambda: driver()),
    }
    while True:
        print("\n==== llama.cpp 双 NVIDIA GPU 管理菜单（Python 版） ====")
        for key, (label, _) in actions.items():
            print(f"{key}) {label}")
        print("0) 退出")
        choice = input("请选择: ").strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if action is None:
            warn("无效选择。")
            continue
        try:
            action[1]()
        except SystemExit as exc:  # die() 抛出的退出不结束菜单
            if exc.code not in (0, None):
                print(f"[ERROR] 操作失败（退出码 {exc.code}）")


def _wizard_entry() -> None:
    cfg, _ = _load_all()
    _config_wizard(cfg)


# -------------------------------------------------------------- B' 新功能 --

profile_app = typer.Typer(help="多配置 profile 管理", no_args_is_help=True)
app.add_typer(profile_app, name="profile")

monitor_app = typer.Typer(help="监控采样与告警", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")


def _profiles_dir() -> Path:
    from .profiles import profiles_dir

    directory = profiles_dir(paths.config_dir())
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@profile_app.command("create")
def profile_create(
    name: str = typer.Argument(..., help="profile 名（字母数字 ._ -）"),
    description: str = typer.Option("", "--desc", help="描述注释"),
) -> None:
    """把当前生效的 server.env 快照为新 profile。"""
    from .profiles import ProfileError, create_profile

    try:
        target = create_profile(paths.config_dir(), name,
                                paths.server_config_file(), description)
    except ProfileError as exc:
        die(str(exc))
    info(f"已创建 profile：{target}")


@profile_app.command("use")
def profile_use(name: str = typer.Argument(..., help="要激活的 profile 名")) -> None:
    """激活 profile：写入 server.env 并记录指针。"""
    from .profiles import ProfileError, use_profile

    try:
        cfg = use_profile(paths.config_dir(), name, paths.server_config_file(),
                          dry_run=STATE.dry_run)
        if not STATE.dry_run:
            errors = cfg.validate()
            if errors:
                die("；".join(errors))  # pragma: no cover — use_profile 内已校验
    except ProfileError as exc:
        die(str(exc))
    info(f"已激活 profile：{name}；重启服务后生效。")


@profile_app.command("list")
def profile_list() -> None:
    """列出所有 profile 及当前激活项。"""
    from .profiles import active_profile, list_profiles

    profiles = list_profiles(paths.config_dir())
    active = active_profile(paths.config_dir())
    if not profiles:
        info("还没有 profile；用 profile create <名字> 快照当前配置。")
        return
    for p in profiles:
        marker = "*" if p.stem == active else " "
        print(f" {marker} {p.stem}")
    if active:
        print(f"\n当前激活：{active}")


@profile_app.command("delete")
def profile_delete(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="跳过确认"),
) -> None:
    """删除 profile（不影响 server.env 当前内容）。"""
    from .profiles import ProfileError, delete_profile

    if not yes and not _confirm(f"确定删除 profile {name}？"):
        warn("已取消。")
        return
    try:
        delete_profile(paths.config_dir(), name)
    except ProfileError as exc:
        die(str(exc))
    info(f"已删除 profile：{name}")


@profile_app.command("show")
def profile_show(name: str = typer.Argument(...)) -> None:
    """查看 profile 内容。"""
    from .profiles import ProfileError, show_profile

    try:
        print(show_profile(paths.config_dir(), name))
    except ProfileError as exc:
        die(str(exc))


def _monitor_env() -> Path:
    return paths.config_dir() / "monitor.env"


@monitor_app.command("run")
def monitor_run(
    once: bool = typer.Option(False, "--once", help="只采样一轮后退出（调试用）"),
) -> None:
    """启动监控采样循环（可注册为 systemd service 长期运行）。"""
    from .monitor import load_monitor_config, run_loop

    try:
        cfg = load_monitor_config(_monitor_env())
    except ValueError as exc:
        die(str(exc))
    db_path = paths.data_home() / "llamacpp" / "metrics.db"
    run_loop(db_path, cfg, once=once or STATE.dry_run)


@monitor_app.command("status")
def monitor_status() -> None:
    """查看最近吞吐与最新 GPU 样本。"""
    from .monitor import connect, latest_tps, recent_alerts

    db_path = paths.data_home() / "llamacpp" / "metrics.db"
    if not db_path.exists():
        info("尚无监控数据；先运行 monitor run。")
        return
    conn = connect(db_path)
    points = latest_tps(conn, limit=10)
    alerts = recent_alerts(conn, limit=10)
    conn.close()
    print("最近吞吐（tok/s）:")
    for ts, tps in reversed(points):
        print(f"  {time.strftime('%H:%M:%S', time.localtime(ts))}  {tps:8.2f}")
    print(f"\n最近告警 {len(alerts)} 条：")
    for ts, rule, level, message, delivered in alerts:
        mark = "[已通知]" if delivered else "[仅记录]"
        print(f"  {time.strftime('%m-%d %H:%M:%S', time.localtime(ts))} {mark} {rule}: {message}")


@monitor_app.command("config")
def monitor_config_command(
    show: bool = typer.Option(False, "--show", help="显示当前监控配置"),
    edit: bool = typer.Option(False, "--edit", help="用编辑器打开 monitor.env"),
) -> None:
    """管理监控配置 monitor.env。"""
    from .monitor import MonitorConfig, load_monitor_config, save_monitor_config

    path = _monitor_env()
    if edit:
        _edit_config(path)
        return
    try:
        cfg = load_monitor_config(path)
    except ValueError:
        cfg = MonitorConfig()
        save_monitor_config(cfg, path)
        info(f"已生成默认配置:{path}")
    for key in ("INTERVAL", "TEMP_MAX", "MEM_PCT_MAX", "HEALTH_FAIL_MAX",
                "ALERT_COOLDOWN", "WEBHOOK_URL", "WEBHOOK_FORMAT",
                "TELEGRAM_CHAT_ID", "PANEL_KEY"):
        value = getattr(cfg, key)
        if key == "PANEL_KEY":
            value = "********" if value else "(未设置)"
        print(f"{key}={value}")


@app.command()
def panel(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址（默认仅本机）"),
    port: int = typer.Option(8199, "--port", min=1, max=65535),
) -> None:
    """启动 Web 管理面板。认证密钥在 monitor.env 的 PANEL_KEY。"""
    from .panel import serve

    if host not in ("127.0.0.1", "localhost"):
        cfg = load_monitor_config_safe()
        if cfg is not None and not cfg.PANEL_KEY:
            warn("面板监听非本机地址但未设置 PANEL_KEY，任何人都可以访问！")
    info(f"面板地址:http://{host}:{port}/")
    try:
        serve(host=host, port=port)
    except OSError as exc:
        die(f"面板启动失败:{exc}")


def load_monitor_config_safe():
    try:
        from .monitor import load_monitor_config

        return load_monitor_config(_monitor_env())
    except Exception:  # noqa: BLE001
        return None


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"llamacpp {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    dry_run: bool = typer.Option(False, "--dry-run", help="显示主要安装/删除操作但不执行"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="禁用交互"),
    verbose: bool = typer.Option(False, "--verbose", help="显示调试信息"),
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="显示版本"
    ),
) -> None:
    STATE.dry_run = dry_run
    STATE.non_interactive = non_interactive
    STATE.verbose = verbose


def main() -> None:
    app()


if __name__ == "__main__":
    main()
