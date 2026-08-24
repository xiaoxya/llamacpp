"""配置模块测试：解析、渲染、校验、legacy 迁移、原子写。"""

from __future__ import annotations

from pathlib import Path

import pytest

from llamacpp.config import (
    BUILD_CONFIG_KEYS,
    BuildConfig,
    ConfigError,
    ServerConfig,
    atomic_write,
    ensure_configs,
    expand_home,
    parse_env_file,
    parse_env_text,
    render_build_config,
    render_server_config,
    save_build_config,
    save_server_config,
)


class TestExpandHome:
    def test_tilde_alone(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        assert expand_home("~") == "/home/tester"

    def test_tilde_prefix(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        assert expand_home("~/models") == "/home/tester/models"

    def test_regression_no_double_expansion(self, monkeypatch):
        """回归：~/ 前缀不得展开成 $HOME/~/（Bash 版曾存在此 bug）。"""
        monkeypatch.setenv("HOME", "/home/tester")
        assert expand_home("~/models/x.gguf") == "/home/tester/models/x.gguf"

    def test_plain_paths_untouched(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        assert expand_home("/opt/data") == "/opt/data"
        assert expand_home("abc") == "abc"


class TestParseEnvText:
    def test_basic_and_comments(self):
        text = "# 注释\n\nMODEL=/tmp/a.gguf\nPORT=9999\n"
        values, warnings = parse_env_text(text, "server")
        assert values["MODEL"] == "/tmp/a.gguf"
        assert values["PORT"] == "9999"
        assert warnings == []

    def test_unknown_key_warned(self):
        values, warnings = parse_env_text("BOGUS=1\nPORT=8080\n", "server")
        assert "BOGUS" not in values
        assert any("BOGUS" in w for w in warnings)

    def test_legacy_key_renamed_outside_server(self):
        values, warnings = parse_env_text("LLAMA_CPP_REF=b456\n", "build")
        assert values.get("LLAMACPP_REF") == "b456"
        assert any("LLAMA_CPP_REF" in w for w in warnings)
        # server 类型不迁移，旧键按未知处理
        values2, _ = parse_env_text("LLAMA_CPP_REF=b456\n", "server")
        assert "LLAMA_CPP_REF" not in values2
        assert "LLAMACPP_REF" not in values2

    def test_crlf_and_missing_file(self, tmp_path):
        values, _ = parse_env_text("PORT=1234\r\n", "server")
        assert values["PORT"] == "1234"
        assert parse_env_file(tmp_path / "nope.env", "server") == ({}, [])


SERVER_SAMPLE = """\
# llama-server startup configuration. Edit values, then run:
#   systemctl --user restart llamacpp.service
# Parsed as data by llamacpp; never sourced as shell code.

# Model
MODEL_DIR=/home/mo/models
MODEL=/home/mo/models/qwen.gguf
MODEL_ALIAS=qwen
MM_PROJ=

# OpenAI-compatible HTTP server
HOST=0.0.0.0
PORT=8080
API_KEY=secret

# Context, concurrency and CPU threads
CTX_SIZE=32768
N_PARALLEL=2
BATCH_SIZE=512
UBATCH_SIZE=256
THREADS=8
THREADS_BATCH=16

# Dual-GPU offload and memory
N_GPU_LAYERS=all
SPLIT_MODE=layer
TENSOR_SPLIT=1,1
MAIN_GPU=0
FIT=true
FIT_TARGET=1536,1536
FLASH_ATTN=auto
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
CUDA_VISIBLE_DEVICES=0,1

# Chat template, reasoning and server features
CONT_BATCHING=true
CACHE_PROMPT=true
JINJA=true
REASONING=auto
REASONING_FORMAT=auto
REASONING_EFFORT=default
REASONING_BUDGET=-1
METRICS=true
WEBUI=true
LOAD_MODE=auto

# Additional native llama-server options. Avoid duplicating managed options above.
EXTRA_ARGS=
"""


class TestServerConfigRoundTrip:
    def test_parse_bash_written_file(self, tmp_path):
        """Bash 版写出的文件必须被 Python 版无损读取。"""
        path = tmp_path / "server.env"
        path.write_text(SERVER_SAMPLE, encoding="utf-8")
        values, warnings = parse_env_file(path, "server")
        cfg = ServerConfig(**values)
        assert warnings == []
        assert cfg.MODEL == "/home/mo/models/qwen.gguf"
        assert cfg.API_KEY == "secret"
        assert cfg.FIT == "true"

    def test_render_idempotent(self, tmp_path):
        path = tmp_path / "server.env"
        path.write_text(SERVER_SAMPLE, encoding="utf-8")
        values, _ = parse_env_file(path, "server")
        cfg = ServerConfig(**values)
        rendered = render_server_config(cfg)
        reparsed, warnings = parse_env_text(rendered, "server")
        assert warnings == []
        expected = {k: getattr(cfg, k) for k in ServerConfig.DEFAULTS}
        assert reparsed == expected

    def test_save_then_load(self, tmp_path):
        target = tmp_path / "cfg" / "server.env"
        cfg = ServerConfig(MODEL_DIR=str(tmp_path / "models"), PORT="9000")
        save_server_config(cfg, target)
        loaded, _ = parse_env_file(target, "server")
        assert loaded["PORT"] == "9000"
        assert target.stat().st_mode & 0o777 == 0o600


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"PORT": "70000"},
            {"PORT": "0"},
            {"FIT": "maybe"},
            {"SPLIT_MODE": "bogus"},
            {"MODEL": "relative/path.gguf"},
            {"CTX_SIZE": "-5"},
            {"MAIN_GPU": "-1"},
            {"CACHE_TYPE_K": "q9_9"},
            {"TENSOR_SPLIT": "1"},
            {"CUDA_VISIBLE_DEVICES": "0;o"},
            {"WEBUI": "TRUE"},
        ],
    )
    def test_invalid_values_detected(self, tmp_path, overrides):
        cfg = ServerConfig(MODEL_DIR=str(tmp_path), **overrides)
        assert cfg.validate(), f"{overrides} 应产生校验错误"

    def test_defaults_pass(self):
        cfg = ServerConfig(MODEL_DIR="/tmp/models")
        assert cfg.validate() == []

    def test_require_valid_raises(self):
        cfg = ServerConfig(MODEL_DIR="/tmp/models", PORT="abc")
        with pytest.raises(ConfigError):
            cfg.require_valid()


class TestBuildConfig:
    def test_defaults_valid(self):
        assert BuildConfig().validate() == []

    def test_invalid(self):
        assert BuildConfig(CUDA_ARCHITECTURES="86,89").validate()
        assert BuildConfig(BUILD_JOBS="-4").validate()
        assert BuildConfig(LLAMACPP_REF="two words").validate()

    def test_legacy_migration(self, tmp_path):
        legacy = tmp_path / "config.env"
        legacy.write_text("LLAMA_CPP_REF=b456\nBUILD_JOBS=4\n", encoding="utf-8")
        values, warnings = parse_env_file(legacy, "legacy")
        build_cfg = BuildConfig(**{k: v for k, v in values.items() if k in BUILD_CONFIG_KEYS})
        assert build_cfg.LLAMACPP_REF == "b456"
        assert build_cfg.BUILD_JOBS == "4"
        assert any("LLAMA_CPP_REF" in w for w in warnings)


class TestEnsureConfigs:
    def test_creates_missing_files(self, tmp_path):
        server = tmp_path / "server.env"
        build = tmp_path / "build.env"
        cfg, bcfg = ensure_configs(server, build)
        assert server.exists() and build.exists()
        loaded, _ = parse_env_file(server, "server")
        assert ServerConfig(**loaded).validate() == []
        assert "master" in build.read_text()

    def test_legacy_migration_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        legacy = tmp_path / "config.env"
        legacy.write_text(
            f"MODEL_DIR={tmp_path}/models\nPORT=8888\nBUILD_JOBS=8\n", encoding="utf-8"
        )
        server = tmp_path / "server.env"
        build = tmp_path / "build.env"
        _, bcfg = ensure_configs(server, build, legacy_path=legacy)
        assert "PORT=8888" in server.read_text()
        assert bcfg.BUILD_JOBS == "8"
        # 迁移后原文件保留
        assert legacy.exists()

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        server = tmp_path / "server.env"
        ensure_configs(server, tmp_path / "build.env", dry_run=True)
        assert not server.exists()
        assert "[DRY-RUN]" in capsys.readouterr().out


class TestAtomicWrite:
    def test_overwrites_and_perms(self, tmp_path):
        target = tmp_path / "x.env"
        atomic_write(target, "A=1\n")
        atomic_write(target, "B=2\n")
        assert target.read_text() == "B=2\n"
        assert target.stat().st_mode & 0o777 == 0o600
