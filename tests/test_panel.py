"""Web 面板测试（fastapi TestClient）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llamacpp.config import ServerConfig, save_server_config
from llamacpp.monitor import MonitorConfig, save_monitor_config
from llamacpp.panel import create_app

HTMX_HEADERS = {"User-Agent": "panel-test"}


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
