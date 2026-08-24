"""Web 管理面板（FastAPI + Jinja2 + HTMX，无前端构建链）。"""

from .app import create_app, render_panel_unit, serve  # noqa: F401
