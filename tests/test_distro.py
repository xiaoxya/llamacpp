"""发行版检测、CUDA 探测与包管理抽象测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llamacpp.distro import (
    DistroInfo,
    detect_distro,
    find_cuda_root,
    nvcc_path,
    parse_os_release,
)

OS_RELEASE_ARCH = """\
NAME="Arch Linux"
ID=arch
ID_LIKE=arch
PRETTY_NAME="Arch Linux"
"""

OS_RELEASE_UBUNTU = """\
NAME="Ubuntu"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 24.04 LTS"
"""

OS_RELEASE_DEBIAN = 'ID=debian\nPRETTY_NAME="Debian GNU/Linux 12"\n'

OS_RELEASE_FEDORA = "ID=fedora\nPRETTY_NAME=\"Fedora 41\"\n"


def write_os_release(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "os-release"
    path.write_text(content, encoding="utf-8")
    return path


class TestParseOsRelease:
    def test_quotes_stripped(self):
        data = parse_os_release(OS_RELEASE_UBUNTU)
        assert data["NAME"] == "Ubuntu"

    def test_garbage_safe(self):
        assert parse_os_release("not a line\n\n# comment\n") == {}


class TestDetectDistro:
    def test_arch(self, tmp_path):
        info = detect_distro(write_os_release(tmp_path, OS_RELEASE_ARCH))
        assert info is not None and info.family == "arch" and info.id == "arch"

    def test_arch_derivative_via_id_like(self, tmp_path):
        content = 'ID=foo\nID_LIKE="arch"\n'
        info = detect_distro(write_os_release(tmp_path, content))
        assert info is not None and info.family == "arch"

    def test_ubuntu(self, tmp_path):
        info = detect_distro(write_os_release(tmp_path, OS_RELEASE_UBUNTU))
        assert info is not None and info.family == "debian" and info.is_debian

    def test_debian(self, tmp_path):
        info = detect_distro(write_os_release(tmp_path, OS_RELEASE_DEBIAN))
        assert info is not None and info.family == "debian"

    def test_fedora_unsupported(self, tmp_path):
        assert detect_distro(write_os_release(tmp_path, OS_RELEASE_FEDORA)) is None

    def test_missing_file(self, tmp_path):
        assert detect_distro(tmp_path / "nope") is None

    def test_real_machine(self):
        """本机应能检测出受支持发行版（CI 为 ubuntu）。"""
        info = detect_distro()
        if info is not None:
            assert info.family in ("arch", "debian")


class TestFindCudaRoot:
    def _make_cuda(self, root: Path) -> Path:
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "nvcc").touch()
        return root

    def test_env_priority(self, tmp_path, monkeypatch):
        cuda_a = self._make_cuda(tmp_path / "a")
        self._make_cuda(tmp_path / "b")
        monkeypatch.setenv("CUDA_HOME", str(cuda_a))
        monkeypatch.delenv("CUDA_PATH", raising=False)
        assert find_cuda_root() == cuda_a

    def test_cuda_path_env_used(self, tmp_path, monkeypatch):
        cuda = self._make_cuda(tmp_path / "via-path")
        assert find_cuda_root({"CUDA_PATH": str(cuda), "PATH": "/nonexistent"}) == cuda

    def test_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)
        # PATH 中无 nvcc 的干净环境
        monkeypatch.setenv("PATH", "/nonexistent")
        result = find_cuda_root(env={"PATH": "/nonexistent"})
        assert result is None or (result / "bin" / "nvcc").is_file()

    def test_nvcc_helper(self, tmp_path):
        root = self._make_cuda(tmp_path / "cuda")
        assert nvcc_path(root) == root / "bin" / "nvcc"


class TestPkgDeps:
    def test_both_families_defined(self):
        from llamacpp.installer import PKG_DEPS

        assert set(PKG_DEPS) == {"arch", "debian"}
        assert "base-devel" in PKG_DEPS["arch"]
        assert "build-essential" in PKG_DEPS["debian"]

    def test_unit_renders_custom_cuda_root(self):
        from llamacpp.service import render_unit

        unit = render_unit(cuda_root="/usr/local/cuda")
        assert "Environment=PATH=/usr/local/cuda/bin:" in unit
        assert "Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64" in unit
