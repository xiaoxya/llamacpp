"""面板应用：仪表盘、服务控制、模型切换、配置编辑、告警历史。

安全约定：
- 默认仅监听 localhost；
- 配置 PANEL_KEY 后启用 Cookie 认证，未登录跳转 /login。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from .. import paths
from .. import service as svc
from ..config import (
    SERVER_CONFIG_KEYS,
    BuildConfig,
    ServerConfig,
    ensure_configs,
    parse_env_file,
    save_server_config,
)
from ..models import human_size, scan_models
from ..monitor import (
    MonitorConfig,
    connect,
    latest_tps,
    load_monitor_config,
    recent_alerts,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
COOKIE_NAME = "panel_token"


def _token_for(key: str) -> str:
    return hashlib.sha256(("llamacpp-panel:" + key).encode()).hexdigest()


def sparkline(points: list[tuple[float, float]], width: int = 480, height: int = 80) -> str:
    """把 (ts, tps) 序列渲染为内联 SVG 折线。"""
    values = [v for _, v in points]
    if len(values) < 2 or max(values) <= 0:
        return "<svg class='spark' viewBox='0 0 480 80'><text x='12' y='46' fill='#8a93a2'>暂无吞吐数据</text></svg>"
    peak = max(values)
    step = width / (len(values) - 1)
    coords = [
        f"{i * step:.1f},{height - 6 - (v / peak) * (height - 16):.1f}"
        for i, v in enumerate(values)
    ]
    return (
        f"<svg class='spark' viewBox='0 0 {width} {height}' preserveAspectRatio='none'>"
        f"<polyline fill='none' stroke='#4f9cf9' stroke-width='2' points='{' '.join(coords)}'/>"
        f"<text x='{width - 8}' y='14' text-anchor='end' fill='#8a93a2'>峰值 {peak:.1f} tok/s</text>"
        "</svg>"
    )


def create_app(
    config_dir: Path | None = None,
    db_path: Path | None = None,
) -> FastAPI:
    config_dir = config_dir or paths.config_dir()
    server_env = config_dir / "server.env"
    build_env = config_dir / "build.env"
    monitor_env = config_dir / "monitor.env"
    db_path = db_path or paths.data_home() / "llamacpp" / "metrics.db"

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="llamacpp panel", docs_url=None, redoc_url=None)

    def load_all() -> tuple[ServerConfig, BuildConfig]:
        cfg, _ = ensure_configs(server_env, build_env, dry_run=False)
        return cfg

    def load_panel_key() -> str:
        try:
            return load_monitor_config(monitor_env).PANEL_KEY
        except Exception:  # noqa: BLE001 — 配置损坏时按未启用处理
            return ""

    def require_auth(request: Request) -> None:
        key = load_panel_key()
        if not key:
            return
        if request.cookies.get(COOKIE_NAME) != _token_for(key):
            raise HTTPException(status_code=303, headers={"Location": "/login"})

    def render(request: Request, template: str, **context) -> HTMLResponse:
        context.setdefault("active_profile", _active_profile_name())
        context.setdefault("panel_key_enabled", bool(load_panel_key()))
        return templates.TemplateResponse(request, template, context)

    def _active_profile_name() -> str | None:
        from ..profiles import active_profile

        try:
            return active_profile(config_dir)
        except Exception:  # noqa: BLE001
            return None

    # ---------------------------------------------------------------- auth --

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return render(request, "login.html", error="")

    @app.post("/login")
    async def login_submit(request: Request, key: str = Form("")):
        expected = load_panel_key()
        if expected and key == expected:
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(COOKIE_NAME, _token_for(expected), httponly=True)
            return response
        return render(request, "login.html", error="密钥错误")

    @app.get("/logout")
    async def logout():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME)
        return response

    # ----------------------------------------------------------- dashboard --

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        require_auth(request)
        cfg = load_all()
        from ..gpu import list_gpus

        gpus = list_gpus()
        service_name = paths.service_file().name
        service_active = svc.is_active(service_name)
        conn = connect(db_path)
        tps_points = latest_tps(conn)
        alerts = recent_alerts(conn, limit=10)
        conn.close()
        return render(
            request, "dashboard.html",
            cfg=cfg, gpus=gpus, service_active=service_active,
            service_name=service_name, spark=Markup(sparkline(tps_points)),
            tps_now=(tps_points[-1][1] if tps_points else None),
            alerts=alerts, model=cfg.MODEL or "(未选择)",
        )

    @app.post("/service/{action}")
    async def service_action(request: Request, action: str):
        require_auth(request)
        name = paths.service_file().name
        if action == "start":
            svc.start(name)
        elif action == "stop":
            svc.stop(name)
        elif action == "restart":
            svc.stop(name)
            svc.start(name)
        else:
            raise HTTPException(404)
        return RedirectResponse("/", status_code=303)

    # ---------------------------------------------------------------- models --

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        require_auth(request)
        cfg = load_all()
        found = scan_models(Path(cfg.MODEL_DIR))
        items = []
        for p in found:
            try:
                size = human_size(p)
            except OSError:
                size = "?"
            items.append({"path": str(p), "name": p.name, "size": size})
        return render(request, "models.html", models=items, current=cfg.MODEL)

    @app.post("/models/select")
    async def models_select(request: Request, path: str = Form(...)):
        require_auth(request)
        cfg = load_all()
        selected = Path(path)
        if not selected.is_file() or selected.suffix.lower() != ".gguf":
            raise HTTPException(400, "无效的 GGUF 路径")
        cfg.MODEL = str(selected)
        if not cfg.MODEL_ALIAS:
            cfg.MODEL_ALIAS = selected.stem
        save_server_config(cfg, server_env)
        return RedirectResponse("/models", status_code=303)

    # ---------------------------------------------------------------- config --

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request):
        require_auth(request)
        cfg = load_all()
        fields = [{"key": k, "value": getattr(cfg, k)} for k in sorted(SERVER_CONFIG_KEYS)]
        errors = request.query_params.get("error", "")
        saved = request.query_params.get("saved") == "1"
        return render(request, "config.html", fields=fields, error=errors, saved=saved,
                      api_key_set=bool(cfg.API_KEY))

    @app.post("/config")
    async def config_save(request: Request):
        require_auth(request)
        form = await request.form()
        values = {}
        for key in SERVER_CONFIG_KEYS:
            if key in form:
                values[key] = str(form[key])
        if "API_KEY" not in values or values["API_KEY"] == "":
            old = load_all()
            values.setdefault("API_KEY", old.API_KEY)  # 留空表示保留旧密钥
        try:
            cfg = ServerConfig(**values)
            errors = cfg.validate()
            if errors:
                raise ValueError("；".join(errors))
            save_server_config(cfg, server_env)
        except Exception as exc:  # noqa: BLE001 — 校验失败回显错误
            from urllib.parse import quote

            return RedirectResponse(f"/config?error={quote(str(exc))}", status_code=303)
        return RedirectResponse("/config?saved=1", status_code=303)

    # ---------------------------------------------------------------- profiles --

    @app.get("/profiles", response_class=HTMLResponse)
    async def profiles_page(request: Request):
        require_auth(request)
        from ..profiles import active_profile, list_profiles

        names = [p.stem for p in list_profiles(config_dir)]
        return render(request, "profiles.html", profiles=names, active=active_profile(config_dir))

    @app.post("/profiles/use")
    async def profiles_use(request: Request, name: str = Form(...)):
        require_auth(request)
        from ..profiles import ProfileError, use_profile

        try:
            use_profile(config_dir, name, server_env)
        except ProfileError as exc:
            raise HTTPException(400, str(exc))
        return RedirectResponse("/profiles", status_code=303)

    # ---------------------------------------------------------------- alerts --

    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request):
        require_auth(request)
        conn = connect(db_path)
        alerts = recent_alerts(conn, limit=100)
        conn.close()
        return render(request, "alerts.html", alerts=alerts)

    return app


def serve(host: str, port: int, config_dir: Path | None = None) -> None:
    import uvicorn

    app = create_app(config_dir=config_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
