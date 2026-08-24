"""Profile 多配置测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from llamacpp.config import ServerConfig, parse_env_file, save_server_config
from llamacpp.profiles import (
    ProfileError,
    active_profile,
    create_profile,
    delete_profile,
    list_profiles,
    use_profile,
)


@pytest.fixture
def setup(tmp_path):
    server_env = tmp_path / "server.env"
    cfg = ServerConfig(MODEL_DIR=str(tmp_path / "models"), PORT="8080",
                       MODEL=str(tmp_path / "models" / "a.gguf"))
    save_server_config(cfg, server_env)
    return tmp_path, server_env


class TestProfileLifecycle:
    def test_create_list_use_delete(self, setup):
        tmp_path, server_env = setup
        create_profile(tmp_path, "fast", server_env)
        create_profile(tmp_path, "long-ctx", server_env)
        assert [p.stem for p in list_profiles(tmp_path)] == ["fast", "long-ctx"]

        # use：写入 server.env 并设置指针
        (tmp_path / "profiles" / "fast.env").write_text(
            (tmp_path / "profiles" / "fast.env").read_text().replace("PORT=8080", "PORT=9000")
        )
        cfg = use_profile(tmp_path, "fast", server_env)
        assert cfg.PORT == "9000"
        reloaded, _ = parse_env_file(server_env, "server")
        assert reloaded["PORT"] == "9000"
        assert active_profile(tmp_path) == "fast"

        delete_profile(tmp_path, "fast")
        assert [p.stem for p in list_profiles(tmp_path)] == ["long-ctx"]
        assert active_profile(tmp_path) is None  # 删除激活项时清空指针

    def test_create_rejects_duplicate_and_bad_name(self, setup):
        tmp_path, server_env = setup
        create_profile(tmp_path, "x", server_env)
        with pytest.raises(ProfileError):
            create_profile(tmp_path, "x", server_env)
        for bad in ("../evil", "", "a/b", ".hidden"):
            with pytest.raises(ProfileError):
                create_profile(tmp_path, bad, server_env)

    def test_use_missing(self, setup):
        tmp_path, server_env = setup
        with pytest.raises(ProfileError):
            use_profile(tmp_path, "nope", server_env)

    def test_use_dry_run_touches_nothing(self, setup):
        tmp_path, server_env = setup
        create_profile(tmp_path, "p1", server_env)
        before = server_env.read_text()
        use_profile(tmp_path, "p1", server_env, dry_run=True)
        assert server_env.read_text() == before
        assert active_profile(tmp_path) is None

    def test_invalid_profile_content_rejected_on_use(self, setup):
        tmp_path, server_env = setup
        profile_file = tmp_path / "profiles" / "broken.env"
        profile_file.parent.mkdir(exist_ok=True)
        profile_file.write_text("PORT=99999\n", encoding="utf-8")
        with pytest.raises(Exception):  # ConfigError 包装的校验失败
            use_profile(tmp_path, "broken", server_env)

    def test_switch_back_and_forth(self, setup):
        """模拟真实使用：两套配置来回切换。"""
        tmp_path, server_env = setup
        create_profile(tmp_path, "big", server_env)
        # 改当前配置再快照为另一套
        values, _ = parse_env_file(server_env, "server")
        values["CTX_SIZE"] = "65536"
        save_server_config(ServerConfig(**values), server_env)
        create_profile(tmp_path, "huge", server_env)

        use_profile(tmp_path, "big", server_env)
        values, _ = parse_env_file(server_env, "server")
        assert values["CTX_SIZE"] == "32768"
        use_profile(tmp_path, "huge", server_env)
        values, _ = parse_env_file(server_env, "server")
        assert values["CTX_SIZE"] == "65536"
