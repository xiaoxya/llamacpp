"""系统安装：pacman 依赖、NVIDIA 驱动、源码克隆、CUDA 编译、安全删除。

与 Bash 版 install/update/driver/uninstall 的行为保持对应。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/ggml-org/llama.cpp.git"
NVCC_PATH = Path("/opt/cuda/bin/nvcc")


class CommandError(Exception):
    pass


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def die(message: str) -> "None":
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def sh(args: list[str], dry_run: bool = False, capture: bool = False, check: bool = True):
    """带 dry-run 支持的子进程执行。"""
    if dry_run:
        print("[DRY-RUN]", *args)
        return subprocess.CompletedProcess(args, 0, "", "")
    result = subprocess.run(
        args,
        check=False,
        text=capture,
        capture_output=capture,
    )
    if check and result.returncode != 0:
        raise CommandError(f"命令失败（退出码 {result.returncode}）：{' '.join(args)}")
    return result


def assert_arch_linux() -> None:
    if not sys.platform.startswith("linux"):
        die("此工具仅支持 Linux。")
    if not Path("/etc/arch-release").exists():
        die("此工具仅支持 Arch Linux 及其直接衍生发行版。")
    import platform

    if platform.machine() != "x86_64":
        die("需要 x86_64。")
    if not shutil.which("pacman"):
        die("未找到 pacman。")


def pacman_install(packages: list[str], non_interactive: bool = False, dry_run: bool = False) -> None:
    args = ["sudo", "pacman", "-S", "--needed"]
    if non_interactive:
        args.append("--noconfirm")
    args += packages
    sh(args, dry_run=dry_run)


def installed_driver_package() -> str | None:
    for package in ("nvidia-open-dkms", "nvidia-open"):
        result = subprocess.run(["pacman", "-Q", package], capture_output=True, check=False)
        if result.returncode == 0:
            return package
    return None


def kernel_header_packages() -> list[str]:
    headers = []
    for kernel in ("linux", "linux-lts", "linux-zen", "linux-hardened"):
        if subprocess.run(["pacman", "-Q", kernel], capture_output=True, check=False).returncode == 0:
            headers.append(f"{kernel}-headers")
    return headers


def manage_driver(
    driver_type: str = "nvidia-open-dkms",
    repair: bool = False,
    non_interactive: bool = False,
    dry_run: bool = False,
) -> None:
    if driver_type not in ("nvidia-open", "nvidia-open-dkms"):
        die(f"驱动类型无效：{driver_type}")
    assert_arch_linux()
    packages = ["nvidia-utils", driver_type]
    if driver_type == "nvidia-open-dkms":
        headers = kernel_header_packages()
        if not headers:
            warn("未识别标准 Arch kernel；请自行安装对应 headers。")
        packages += ["dkms", *headers]
    current = installed_driver_package()
    info(f"当前驱动包:{current or '无'}；目标：{driver_type}")
    if repair:
        args = ["sudo", "pacman", "-S"]
        if non_interactive:
            args.append("--noconfirm")
        sh([*args, *packages], dry_run=dry_run)
    else:
        pacman_install(packages, non_interactive=non_interactive, dry_run=dry_run)
    if shutil.which("mkinitcpio"):
        sh(["sudo", "mkinitcpio", "-P"], dry_run=dry_run, check=False)
    warn("驱动变更后必须重启；工具不会自动重启。")


BUILD_DEPS = ["base-devel", "cmake", "ninja", "git", "ccache", "cuda", "openssl", "curl", "python"]


def install_build_dependencies(non_interactive: bool = False, dry_run: bool = False) -> None:
    assert_arch_linux()
    info("安装 CUDA Toolkit 和 llama.cpp 构建依赖……")
    pacman_install(BUILD_DEPS, non_interactive=non_interactive, dry_run=dry_run)


def available_cpu_threads() -> int:
    try:
        count = os.cpu_count() or 1
    except Exception:  # pragma: no cover
        count = 1
    return max(count, 1)


def resolved_build_jobs(build_jobs: str) -> int:
    return available_cpu_threads() if build_jobs == "auto" else int(build_jobs)


def source_tree_is_clean(source_dir: Path) -> bool:
    def quiet(*args: str) -> bool:
        return subprocess.run(
            ["git", "-C", str(source_dir), *args],
            capture_output=True,
            check=False,
        ).returncode == 0

    return quiet("diff", "--quiet", "--ignore-submodules", "--") and quiet(
        "diff", "--cached", "--quiet", "--ignore-submodules", "--"
    )


def prepare_source(source_dir: Path, ref: str = "master", dry_run: bool = False) -> None:
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (source_dir / ".git").exists():
        info("克隆 llama.cpp 官方仓库……")
        sh(["git", "clone", "--filter=blob:none", REPOSITORY_URL, str(source_dir)], dry_run=dry_run)
    else:
        if not source_tree_is_clean(source_dir):
            die(f"源码目录有本地修改，拒绝覆盖:{source_dir}")
        info("获取 llama.cpp 更新……")
        sh(["git", "-C", str(source_dir), "fetch", "--tags", "--prune", "origin"], dry_run=dry_run)
    if ref == "master":
        sh(["git", "-C", str(source_dir), "checkout", "master"], dry_run=dry_run)
        sh(["git", "-C", str(source_dir), "pull", "--ff-only", "origin", "master"], dry_run=dry_run)
    else:
        sh(["git", "-C", str(source_dir), "checkout", "--detach", ref], dry_run=dry_run)


def build_llama_cpp(
    source_dir: Path,
    build_dir: Path,
    cuda_architectures: str,
    build_jobs: str,
    dry_run: bool = False,
) -> None:
    if not dry_run and not NVCC_PATH.is_file():
        die(f"未找到 {NVCC_PATH}；请确认 Arch cuda 包安装成功。")
    env = dict(os.environ)
    env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"
    env["CUDACXX"] = str(NVCC_PATH)
    jobs = resolved_build_jobs(build_jobs)
    sm_list = ",".join(f"sm_{a}" for a in cuda_architectures.split(";"))
    info(f"配置 CUDA 构建：{sm_list}，并行任务 {jobs}（BUILD_JOBS={build_jobs}）……")
    configure = [
        "cmake", "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_NATIVE=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DLLAMA_CURL=ON",
        "-DLLAMA_OPENSSL=ON",
    ]
    build_cmd = [
        "cmake", "--build", str(build_dir),
        "--target", "llama-server", "llama-cli", "llama-bench",
        "--parallel", str(jobs),
    ]
    if dry_run:
        print("[DRY-RUN]", *configure)
        print("[DRY-RUN]", *build_cmd)
        return
    for args in (configure, build_cmd):
        result = subprocess.run(args, env=env, check=False)
        if result.returncode != 0:
            raise CommandError(f"命令失败（退出码 {result.returncode}）：{' '.join(args)}")


def safe_remove_tree(target: Path, home: Path | None = None, dry_run: bool = False) -> None:
    """仅允许删除家目录下的路径，拒绝越界。"""
    home = home or Path.home()
    resolved = Path(os.path.realpath(target))
    if resolved != home and not resolved.is_relative_to(home):
        die(f"拒绝删除不安全路径:{resolved}")
    if dry_run:
        print(f"[DRY-RUN] rm -rf {resolved}")
        return
    shutil.rmtree(resolved)


def git_short_commit(source_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None
