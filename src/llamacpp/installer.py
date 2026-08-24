"""系统安装：依赖包、NVIDIA 驱动、源码克隆、CUDA 编译、安全删除。

支持 Arch 系（pacman）与 Debian/Ubuntu 系（apt）；CUDA 路径自动探测，
不写死发行版路径。与 Bash 版 install/update/driver/uninstall 行为对应。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .distro import current_distro, find_cuda_root, nvcc_path

REPOSITORY_URL = "https://github.com/ggml-org/llama.cpp.git"

# 各家族构建依赖；Debian 系的 cuda-toolkit 需 NVIDIA 官方 repo，见提示
PKG_DEPS: dict[str, list[str]] = {
    "arch": ["base-devel", "cmake", "ninja", "git", "ccache", "cuda",
             "openssl", "curl", "python"],
    "debian": ["build-essential", "cmake", "ninja-build", "git", "ccache",
               "curl", "python3", "libcurl4-openssl-dev", "libssl-dev"],
}


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


def assert_supported_linux() -> tuple[str, object]:
    """校验当前为受支持的 Linux 发行版，返回 (family, distro)。"""
    if not sys.platform.startswith("linux"):
        die("此工具仅支持 Linux。")
    if platform.machine() not in ("x86_64", "aarch64"):
        die(f"暂不支持架构：{platform.machine()}（需要 x86_64 或 aarch64）。")
    distro = current_distro()
    if distro is None:
        die("未识别受支持的发行版（支持 Arch 系与 Debian/Ubuntu 系）。")
    return distro.family, distro


# 兼容旧名：内部逻辑已改为多发行版
def assert_arch_linux() -> None:
    family, distro = assert_supported_linux()
    info(f"检测到发行版：{distro.pretty_name}（{family} 系）")


def _pkg_install(packages: list[str], non_interactive: bool = False,
                 dry_run: bool = False) -> None:
    """按发行版家族安装系统包。"""
    family, _ = assert_supported_linux()
    if family == "arch":
        args = ["sudo", "pacman", "-S", "--needed"]
        if non_interactive:
            args.append("--noconfirm")
        sh([*args, *packages], dry_run=dry_run)
        return

    sh(["sudo", "apt-get", "update"], dry_run=dry_run)
    cmd = ["sudo", "apt-get", "install", "-y", *packages]
    if dry_run:
        print("[DRY-RUN]", *cmd)
        return
    env = dict(os.environ)
    if non_interactive:
        env["DEBIAN_FRONTEND"] = "noninteractive"
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise CommandError(f"命令失败（退出码 {result.returncode}）：{' '.join(cmd)}")


def pacman_install(packages: list[str], non_interactive: bool = False,
                   dry_run: bool = False) -> None:
    """兼容别名：按当前发行版安装包。"""
    _pkg_install(packages, non_interactive=non_interactive, dry_run=dry_run)


def _dpkg_has(package: str) -> bool:
    return subprocess.run(["dpkg", "-l", package], capture_output=True,
                          check=False).returncode == 0


def installed_driver_package() -> str | None:
    distro = current_distro()
    if distro and distro.is_debian:
        for package in ("nvidia-driver-550", "nvidia-driver-535",
                        "nvidia-open", "nvidia-driver"):
            if _dpkg_has(package):
                return package
        return None
    for package in ("nvidia-open-dkms", "nvidia-open"):
        result = subprocess.run(["pacman", "-Q", package],
                                capture_output=True, check=False)
        if result.returncode == 0:
            return package
    return None


def kernel_header_packages() -> list[str]:
    """返回当前系统应安装的内核头文件包名（按家族）。"""
    distro = current_distro()
    if distro and distro.is_debian:
        release = subprocess.run(["uname", "-r"], capture_output=True,
                                 text=True).stdout.strip()
        return [f"linux-headers-{release}"] if release else []
    headers = []
    for kernel in ("linux", "linux-lts", "linux-zen", "linux-hardened"):
        if subprocess.run(["pacman", "-Q", kernel],
                          capture_output=True, check=False).returncode == 0:
            headers.append(f"{kernel}-headers")
    return headers


def manage_driver(
    driver_type: str = "auto",
    repair: bool = False,
    non_interactive: bool = False,
    dry_run: bool = False,
) -> None:
    """安装/修复 NVIDIA 驱动。

    - Arch 系：nvidia-open / nvidia-open-dkms + 内核头文件 + mkinitcpio
    - Debian/Ubuntu：auto 走 ubuntu-drivers 自动匹配；也可直接传 apt 包名
      （如 nvidia-driver-550）。Debian 需先启用 non-free 仓库。
    """
    family, distro = assert_supported_linux()

    if family == "debian":
        current = installed_driver_package()
        info(f"当前驱动包:{current or '无'}；目标：{driver_type}")
        if driver_type == "auto":
            if shutil.which("ubuntu-drivers"):
                sh(["sudo", "ubuntu-drivers", "autoinstall"],
                   dry_run=dry_run, check=False)
                warn("驱动变更后必须重启；工具不会自动重启。")
                return
            die(
                "未找到 ubuntu-drivers。请手动指定 apt 驱动包名，例如：\n"
                "  llamacpp-py driver --type nvidia-driver-550\n"
                "Ubuntu 推荐 sudo ubuntu-drivers autoinstall；"
                "Debian 需先启用 non-free 仓库。"
            )
        _pkg_install([driver_type, *kernel_header_packages()],
                     non_interactive=non_interactive, dry_run=dry_run)
        if shutil.which("update-initramfs"):
            sh(["sudo", "update-initramfs", "-u"], dry_run=dry_run, check=False)
        warn("驱动变更后必须重启；工具不会自动重启。")
        return

    # Arch 家族
    if driver_type == "auto":
        driver_type = "nvidia-open-dkms"
    if driver_type not in ("nvidia-open", "nvidia-open-dkms"):
        die(f"驱动类型无效:{driver_type}（可选 nvidia-open / nvidia-open-dkms）")
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
        _pkg_install(packages, non_interactive=non_interactive, dry_run=dry_run)
    if shutil.which("mkinitcpio"):
        sh(["sudo", "mkinitcpio", "-P"], dry_run=dry_run, check=False)
    warn("驱动变更后必须重启；工具不会自动重启。")


def install_build_dependencies(non_interactive: bool = False, dry_run: bool = False) -> None:
    family, distro = assert_supported_linux()
    info(f"[{distro.pretty_name}] 安装构建依赖……")
    deps = list(PKG_DEPS[family])
    if family == "debian":
        # cuda-toolkit 在 Debian/Ubuntu 需 NVIDIA 官方 repo；失败时降级并给指引
        try:
            _pkg_install([*deps, "nvidia-cuda-toolkit"],
                         non_interactive=non_interactive, dry_run=dry_run)
            return
        except CommandError:
            warn("apt 安装 nvidia-cuda-toolkit 失败——继续安装其余依赖；")
            warn("CUDA Toolkit 请按官方指南安装：https://developer.nvidia.com/cuda-downloads")
    _pkg_install(deps, non_interactive=non_interactive, dry_run=dry_run)


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
    cuda_root = find_cuda_root()
    compiler = nvcc_path(cuda_root) if cuda_root else None
    if not dry_run and (cuda_root is None or not compiler.is_file()):
        die(
            "未找到 CUDA Toolkit（nvcc）。请安装后重试：\n"
            "  Ubuntu：sudo apt install nvidia-cuda-toolkit 或官方 repo\n"
            "  Arch：sudo pacman -S cuda\n"
            "  自定义路径：export CUDA_HOME=/path/to/cuda"
        )
    env = dict(os.environ)
    env["PATH"] = f"{cuda_root}/bin:{env.get('PATH', '')}"
    env["CUDACXX"] = str(compiler)
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
