"""模型扫描、GPU 解析、service unit 与 CLI 冒烟测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from llamacpp.gpu import parse_gpu_table
from llamacpp.models import scan_models
from llamacpp.service import render_unit

SRC = str(Path(__file__).resolve().parents[1] / "src")


class TestScanModels:
    def test_shard_and_mmproj_filtering(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        names = [
            "model-a.gguf",
            "mmproj-model.gguf",
            "qwen-00001-of-00003.gguf",
            "qwen-00002-of-00003.gguf",
            "notes.txt",
            "UPPER.GGUF",
        ]
        for name in names:
            (root / name).touch()
        found = [p.name for p in scan_models(root)]
        assert "model-a.gguf" in found
        assert "UPPER.GGUF" in found
        assert "mmproj-model.gguf" not in found
        assert "qwen-00001-of-00003.gguf" in found
        assert "qwen-00002-of-00003.gguf" not in found
        assert "notes.txt" not in found

    def test_missing_dir_returns_empty(self, tmp_path):
        assert scan_models(tmp_path / "nope") == []


class TestGpuParser:
    SAMPLE = """\
0, NVIDIA GeForce RTX 4070 SUPER, 12282, 12000, 44, 575.57.08
1, NVIDIA GeForce RTX 3060, 12288, 11800, 51, 575.57.08
"""

    def test_parse(self):
        gpus = parse_gpu_table(self.SAMPLE)
        assert len(gpus) == 2
        assert gpus[0].name == "NVIDIA GeForce RTX 4070 SUPER"
        assert gpus[0].memory_total_mib == 12282
        assert gpus[1].temperature == 51
        assert gpus[1].driver_version == "575.57.08"


def test_render_unit_matches_bash_layout():
    unit = render_unit()
    assert "[Unit]" in unit
    assert "ExecStart=%h/.local/bin/llamacpp-start" in unit
    assert "WantedBy=default.target" in unit

def test_render_unit_cuda_root_parameterized():
    from llamacpp.service import render_unit as r

    # Arch 惯例路径与 Ubuntu 官方 repo 路径都应可用
    for cuda in ("/opt/cuda", "/usr/local/cuda"):
        unit = r(cuda_root=cuda)
        assert f"Environment=PATH={cuda}/bin:" in unit
        assert f"Environment=LD_LIBRARY_PATH={cuda}/lib64" in unit


class TestCliSmoke:
    def run_cli(self, *args: str, home: Path | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        if home is not None:
            # 隔离所有影响路径解析的环境变量（CI runner 可能预置 XDG_*）
            env["HOME"] = str(home)
            env["XDG_CONFIG_HOME"] = str(home / ".config")
            env["XDG_DATA_HOME"] = str(home / ".local/share")
        return subprocess.run(
            [sys.executable, "-m", "llamacpp", *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def test_version(self):
        result = self.run_cli("--version")
        assert result.returncode == 0
        assert result.stdout.startswith("llamacpp ")

    def test_help_exit_zero(self):
        assert self.run_cli("--help").returncode == 0

    def test_config_show_creates_defaults(self, tmp_path):
        result = self.run_cli("config", "--show", home=tmp_path)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / ".config/llamacpp/server.env").exists()
        assert (tmp_path / ".config/llamacpp/build.env").exists()
        assert "MODEL_DIR=" in result.stdout
        # API_KEY 默认为空，显示为未设置而不是泄漏
        assert "API_KEY=(未设置)" in result.stdout

    def test_config_set_and_reload(self, tmp_path):
        first = self.run_cli("config", "PORT", "9999", home=tmp_path)
        assert first.returncode == 0, first.stderr
        content = (tmp_path / ".config/llamacpp/server.env").read_text()
        assert "PORT=9999" in content
        second = self.run_cli("config", "--show", home=tmp_path)
        assert "PORT=9999" in second.stdout

    def test_config_rejects_bad_value(self, tmp_path):
        result = self.run_cli("config", "PORT", "70000", home=tmp_path)
        assert result.returncode != 0

    def test_config_rejects_unknown_key(self, tmp_path):
        result = self.run_cli("config", "NOT_A_KEY", "x", home=tmp_path)
        assert result.returncode != 0
