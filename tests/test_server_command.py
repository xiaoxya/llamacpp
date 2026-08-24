"""命令构建与健康检查测试：与 Bash 版参数一一对应。"""

from __future__ import annotations

import pytest

from llamacpp.config import ServerConfig
from llamacpp.server import (
    build_bench_command,
    build_server_command,
    health_url,
    parse_extra_args,
)

BIN = Path_ = "/opt/llamacpp/build/bin/llama-server"


def make_cfg(**overrides) -> ServerConfig:
    values = {"MODEL_DIR": "/tmp/models", "MODEL": "/tmp/models/m.gguf"}
    values.update(overrides)
    return ServerConfig(**values)


class TestServerCommand:
    def test_default_shape(self):
        cfg = make_cfg()
        argv = build_server_command(cfg, BIN)
        assert argv[0] == BIN
        for flag, value in [
            ("--model", cfg.MODEL),
            ("--host", "0.0.0.0"),
            ("--port", "8080"),
            ("--ctx-size", "32768"),
            ("--parallel", "2"),
            ("--n-gpu-layers", "all"),
            ("--split-mode", "layer"),
            ("--tensor-split", "1,1"),
            ("--flash-attn", "auto"),
            ("--cache-type-k", "q8_0"),
            ("--load-mode", "auto"),
            ("--reasoning", "auto"),
        ]:
            idx = argv.index(flag)
            assert argv[idx + 1] == value, f"{flag} 参数值不符"
        assert "--log-timestamps" in argv
        assert "--fit" in argv and argv[argv.index("--fit") + 1] == "on"
        assert argv[argv.index("--fit-target") + 1] == "1536,1536"
        assert "--cont-batching" in argv
        assert "--cache-prompt" in argv
        assert "--jinja" in argv
        assert "--webui" in argv
        assert "--metrics" in argv

    def test_optional_flags_included(self):
        cfg = make_cfg(
            MODEL_ALIAS="qwen", MM_PROJ="/tmp/mm.gguf", API_KEY="secret",
            REASONING_FORMAT="deepseek", REASONING_EFFORT="high",
            REASONING_BUDGET="1024",
        )
        argv = build_server_command(cfg, BIN)
        assert argv[argv.index("--alias") + 1] == "qwen"
        assert argv[argv.index("--mmproj") + 1] == "/tmp/mm.gguf"
        assert argv[argv.index("--api-key") + 1] == "secret"
        assert argv[argv.index("--reasoning-format") + 1] == "deepseek"
        assert argv[argv.index("--reasoning-effort") + 1] == "high"
        assert argv[argv.index("--reasoning-budget") + 1] == "1024"

    def test_optional_flags_absent_when_defaults(self):
        argv = build_server_command(make_cfg(), BIN)
        for flag in ("--alias", "--mmproj", "--api-key",
                     "--reasoning-format", "--reasoning-effort", "--reasoning-budget"):
            assert flag not in argv

    @pytest.mark.parametrize(
        ("key", "off_flag", "on_flag"),
        [
            ("CONT_BATCHING", "--no-cont-batching", "--cont-batching"),
            ("CACHE_PROMPT", "--no-cache-prompt", "--cache-prompt"),
            ("JINJA", "--no-jinja", "--jinja"),
            ("WEBUI", "--no-webui", "--webui"),
        ],
    )
    def test_bool_switch_pairs(self, key, on_flag, off_flag):
        on_argv = build_server_command(make_cfg(**{key: "true"}), BIN)
        off_argv = build_server_command(make_cfg(**{key: "false"}), BIN)
        assert on_flag in on_argv and off_flag not in on_argv
        assert off_flag in off_argv and on_flag not in off_argv

    def test_fit_off(self):
        argv = build_server_command(make_cfg(FIT="false"), BIN)
        assert argv[argv.index("--fit") + 1] == "off"
        assert "--fit-target" not in argv

    def test_metrics_off(self):
        assert "--metrics" not in build_server_command(make_cfg(METRICS="false"), BIN)

    def test_extra_args_appended(self):
        argv = build_server_command(make_cfg(EXTRA_ARGS='--verbose "--top-p 0.9"'), BIN)
        assert argv[-2:] == ["--verbose", "--top-p 0.9"]


class TestBenchCommand:
    def test_shape(self):
        cfg = make_cfg(N_GPU_LAYERS="24")
        argv = build_bench_command(cfg, "/bin/llama-bench")
        assert argv[0] == "/bin/llama-bench"
        assert "-ngl" in argv and argv[argv.index("-ngl") + 1] == "24"
        assert "-ts" in argv and argv[argv.index("-ts") + 1] == "1/1"
        assert "--progress" in argv

    def test_ngl_all_maps_to_minus_one(self):
        argv = build_bench_command(make_cfg(N_GPU_LAYERS="all"), "/bin/llama-bench")
        assert argv[argv.index("-ngl") + 1] == "-1"


class TestParseExtraArgs:
    def test_empty(self):
        assert parse_extra_args("") == []

    def test_quoting(self):
        assert parse_extra_args('a b "c d"') == ["a", "b", "c d"]


class TestHealthUrl:
    @pytest.mark.parametrize(
        ("host", "expected_prefix"),
        [
            ("0.0.0.0", "http://127.0.0.1:8080/health"),
            ("", "http://127.0.0.1:8080/health"),
            ("::", "http://[::1]:9000/health"),
            ("[::]", "http://[::1]:8080/health"),
            ("192.168.1.5", "http://192.168.1.5:8080/health"),
            ("::1", "http://[::1]:8080/health"),
        ],
    )
    def test_mapping(self, host, expected_prefix):
        port = "9000" if host == "::" else "8080"
        assert health_url(host, port) == expected_prefix
