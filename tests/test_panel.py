"""Web 面板测试（fastapi TestClient）。"""

from __future__ import annotations

import hashlib
import os
import subprocess as sp
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llamacpp.config import ServerConfig, save_server_config
from llamacpp.monitor import MonitorConfig, save_monitor_config
from llamacpp.panel import create_app

HTMX_HEADERS = {"User-Agent": "panel-test"}

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def env(tmp_path):
    server_env = tmp_path / "server.env"
    save_server_config(
        ServerConfig(MODEL_DIR=str(tmp_path / "models"), PORT="8080"), server_env
    )
    db = tmp_path / "metrics.db"
    return tmp_path, server_env, db


def make_client(env, panel_key: str = "") -> TestClient:
    tmp_path, _server_env, db = env
    if panel_key:
        save_monitor_config(MonitorConfig(PANEL_KEY=panel_key), tmp_path / "monitor.env")
    app = create_app(config_dir=tmp_path, db_path=db)
    return TestClient(app, follow_redirects=False)


class TestDashboard:
    def test_dashboard_renders(self, env):
        client = make_client(env)
        resp = client.get("/", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "llamacpp" in resp.text
        assert "仪表盘" in resp.text

    def test_service_stop_action(self, env, monkeypatch):
        called = {}
        import llamacpp.panel.app as panel_app

        monkeypatch.setattr(panel_app.svc, "stop",
                            lambda name, dry_run=False: called.setdefault("stop", name))
        client = make_client(env)
        resp = client.post("/service/stop", headers=HTMX_HEADERS)
        assert resp.status_code == 303
        assert called.get("stop")


class TestModelsPage:
    def test_list_and_select(self, env):
        tmp_path, _server_env, _db = env
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "a.gguf").touch()
        (models_dir / "b.gguf").touch()
        client = make_client(env)
        page = client.get("/models", headers=HTMX_HEADERS)
        assert "a.gguf" in page.text and "b.gguf" in page.text

        resp = client.post("/models/select", data={"path": str(models_dir / "b.gguf")},
                           headers=HTMX_HEADERS)
        assert resp.status_code == 303
        content = (tmp_path / "server.env").read_text()
        assert f"MODEL={models_dir / 'b.gguf'}" in content

    def test_select_rejects_non_gguf(self, env, tmp_path=None):
        tmp_path = env[0]
        fake = tmp_path / "evil.txt"
        fake.touch()
        client = make_client(env)
        resp = client.post("/models/select", data={"path": str(fake)},
                           headers=HTMX_HEADERS)
        assert resp.status_code == 400


class TestConfigPage:
    def test_view_and_save(self, env):
        tmp_path, server_env, _db = env
        client = make_client(env)
        page = client.get("/config", headers=HTMX_HEADERS)
        assert "CTX_SIZE" in page.text

        resp = client.post("/config", data={"PORT": "9999"}, headers=HTMX_HEADERS)
        assert resp.status_code == 303
        values, _ = (lambda p: __import__("llamacpp.config", fromlist=["parse_env_file"])
                     .parse_env_file(p, "server"))(server_env)
        assert values["PORT"] == "9999"

    def test_invalid_save_shows_error(self, env):
        client = make_client(env)
        resp = client.post("/config", data={"PORT": "70000"},
                           headers=HTMX_HEADERS, follow_redirects=True)
        assert resp.status_code == 200


class TestAlertsPage:
    def test_empty_alerts(self, env):
        client = make_client(env)
        resp = client.get("/alerts", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "暂无告警" in resp.text


class TestAuth:
    def test_key_required_when_configured(self, env):
        client = make_client(env, panel_key="sekrit")
        # 未登录 → 303 跳转登录页
        resp = client.get("/", headers=HTMX_HEADERS)
        assert resp.status_code in (303, 307)
        assert "/login" in resp.headers.get("location", "")

    def test_login_flow(self, env):
        client = make_client(env, panel_key="sekrit")
        bad = client.post("/login", data={"key": "wrong"}, headers=HTMX_HEADERS)
        assert bad.status_code == 200 and "密钥错误" in bad.text

        good = client.post("/login", data={"key": "sekrit"}, headers=HTMX_HEADERS)
        assert good.status_code == 303
        token = hashlib.sha256(b"llamacpp-panel:sekrit").hexdigest()
        client.cookies.set("panel_token", token)
        assert client.get("/", headers=HTMX_HEADERS).status_code == 200

    def test_no_key_open_access(self, env):
        client = make_client(env)
        assert client.get("/", headers=HTMX_HEADERS).status_code == 200


class TestProfilesPage:
    def test_profiles_page_lists(self, env):
        from llamacpp.profiles import create_profile

        tmp_path, server_env, _db = env
        create_profile(tmp_path, "p1", server_env)
        client = make_client(env)
        resp = client.get("/profiles", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        assert "p1" in resp.text

    def test_use_profile_from_panel(self, env):
        from llamacpp.profiles import create_profile

        tmp_path, server_env, _db = env
        create_profile(tmp_path, "p1", server_env)
        client = make_client(env)
        resp = client.post("/profiles/use", data={"name": "p1"}, headers=HTMX_HEADERS)
        assert resp.status_code == 303


class TestPanelServiceCommands:
    def test_render_unit(self):
        from llamacpp.panel import render_panel_unit

        unit = render_panel_unit("/home/x/.local/bin/llamacpp-py", "0.0.0.0", 8199)
        assert "ExecStart=/home/x/.local/bin/llamacpp-py panel serve --host 0.0.0.0 --port 8199" in unit
        assert "Restart=on-failure" in unit
        assert "[Install]" in unit

    def test_install_writes_unit(self, env, monkeypatch):
        tmp_path, _server_env, _db = env
        import llamacpp.panel.app as panel_app
        import llamacpp.service as svc

        monkeypatch.setattr(svc, "daemon_reload", lambda dry_run=False: None)
        monkeypatch.setattr(svc, "enable", lambda name, dry_run=False: None)
        monkeypatch.setattr(panel_app.svc, "daemon_reload", lambda dry_run=False: None)
        monkeypatch.setattr(panel_app.svc, "enable", lambda name, dry_run=False: None)

        client = make_client(env)  # 仅确保 app 可构建；CLI 走 subprocess
        del client
        from llamacpp.paths import systemd_user_dir
        import os
        import subprocess as sp

        env_vars = dict(os.environ)
        env_vars.update({
            "PYTHONPATH": SRC,
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / ".config"),
            "XDG_DATA_HOME": str(tmp_path / ".local/share"),
        })
        result = sp.run(
            [sys.executable, "-m", "llamacpp", "panel", "install",
             "--host", "127.0.0.1", "--no-start"],
            capture_output=True, text=True, env=env_vars, check=False,
        )
        assert result.returncode == 0, result.stderr
        unit = tmp_path / ".config/systemd/user/llamacpp-panel.service"
        assert unit.exists()
        assert "panel serve --host 127.0.0.1 --port 8199" in unit.read_text()

    def test_install_lan_requires_key(self, env):
        import os
        import subprocess as sp

        tmp_path = env[0]
        env_vars = dict(os.environ)
        env_vars.update({
            "PYTHONPATH": SRC,
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / ".config"),
            "XDG_DATA_HOME": str(tmp_path / ".local/share"),
        })
        # 未设置 PANEL_KEY，监听 0.0.0.0 必须被拒绝且不产生 unit 文件
        result = sp.run(
            [sys.executable, "-m", "llamacpp", "panel", "install",
             "--host", "0.0.0.0", "--no-start"],
            capture_output=True, text=True, env=env_vars, check=False,
        )
        assert result.returncode != 0
        assert "PANEL_KEY" in (result.stderr + result.stdout)
        assert not (tmp_path / ".config/systemd/user/llamacpp-panel.service").exists()

    def test_cli_panel_help(self, env):
        import os
        import subprocess as sp

        tmp_path = env[0]
        env_vars = dict(os.environ)
        env_vars["PYTHONPATH"] = SRC
        env_vars["HOME"] = str(tmp_path)
        result = sp.run([sys.executable, "-m", "llamacpp", "panel", "--help"],
                        capture_output=True, text=True, env=env_vars, check=False)
        assert result.returncode == 0
        for cmd in ("serve", "install", "start", "stop", "restart", "status", "logs"):
            assert cmd in result.stdout


class TestLoginWithQuotedKey:
    """端到端：monitor.env 带引号的 PANEL_KEY 也能正常登录。"""

    def test_login_success_with_quoted_config(self, env):
        tmp_path = env[0]
        (tmp_path / "monitor.env").write_text(
            'PANEL_KEY="sekrit"\n', encoding="utf-8"
        )
        client = make_client(env)
        resp = client.post("/login", data={"key": " sekrit "}, headers=HTMX_HEADERS)
        assert resp.status_code == 303
        import hashlib

        token = hashlib.sha256(b"llamacpp-panel:sekrit").hexdigest()
        client.cookies.set("panel_token", token)
        assert client.get("/", headers=HTMX_HEADERS).status_code == 200


class TestApiLive:
    def test_open_access_returns_payload(self, env):
        import json

        client = make_client(env)
        resp = client.get("/api/live", headers=HTMX_HEADERS)
        assert resp.status_code == 200
        data = json.loads(resp.text)
        for key in ("service_active", "model", "gpus", "tps_now", "series", "alerts"):
            assert key in data

    def test_auth_required_when_key_set(self, env):
        client = make_client(env, panel_key="k1")
        resp = client.get("/api/live", headers=HTMX_HEADERS)
        assert resp.status_code == 401
