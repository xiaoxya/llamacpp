"""监控采样与告警。

- 采集：nvidia-smi（显存/温度/利用率）+ llama-server /metrics（token 计数）
- 存储：SQLite（~/.local/share/llamacpp/metrics.db）
- 告警：阈值触发 → 面板内记录 + 日志 + Webhook（generic/wecom/dingtalk/
  telegram/bark 格式适配）
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import atomic_write
from .server import health_url, http_get

DEFAULT_DB_PATH = Path.home() / ".local/share/llamacpp/metrics.db"

MONITOR_KEYS = (
    "INTERVAL", "TEMP_MAX", "MEM_PCT_MAX", "HEALTH_FAIL_MAX",
    "WEBHOOK_URL", "WEBHOOK_FORMAT", "TELEGRAM_CHAT_ID", "API_KEY", "PORT",
    "HOST", "ALERT_COOLDOWN", "PANEL_KEY",
)


@dataclass
class MonitorConfig:
    INTERVAL: int = 10            # 采样间隔（秒）
    TEMP_MAX: int = 85            # GPU 温度告警阈值（℃）
    MEM_PCT_MAX: int = 95         # 显存使用率告警阈值（%）
    HEALTH_FAIL_MAX: int = 3      # 连续 health 失败次数告警
    ALERT_COOLDOWN: int = 300     # 同类告警冷却（秒）
    WEBHOOK_URL: str = ""
    WEBHOOK_FORMAT: str = "generic"   # generic|wecom|dingtalk|telegram|bark
    TELEGRAM_CHAT_ID: str = ""
    API_KEY: str = ""
    HOST: str = "127.0.0.1"
    PORT: str = "8080"
    PANEL_KEY: str = ""   # 面板登录密钥；为空则不启用认证（默认仅监听 localhost）

    def __post_init__(self) -> None:
        # env 文件读出的数值键统一转为 int
        for key in ("INTERVAL", "TEMP_MAX", "MEM_PCT_MAX", "HEALTH_FAIL_MAX", "ALERT_COOLDOWN"):
            value = getattr(self, key)
            try:
                setattr(self, key, int(value))
            except (TypeError, ValueError):
                pass  # 留给 validate() 报告

    def validate(self) -> list[str]:
        errors = []
        if self.INTERVAL < 2:
            errors.append("INTERVAL 必须 >= 2 秒。")
        if self.WEBHOOK_FORMAT not in ("generic", "wecom", "dingtalk", "telegram", "bark"):
            errors.append("WEBHOOK_FORMAT 无效。")
        if not (0 < int(self.TEMP_MAX) <= 120):
            errors.append("TEMP_MAX 无效。")
        if not (0 < int(self.MEM_PCT_MAX) <= 100):
            errors.append("MEM_PCT_MAX 无效。")
        return errors


def _unquote(value: str) -> str:
    """去掉值两侧的成对引号（用户常按 shell 习惯写 KEY="xxx"）。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _read_monitor_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key in MONITOR_KEYS:
                values[key] = _unquote(value)
    return values


def load_monitor_config(path: Path, server_env_path: Path | None = None
                        ) -> MonitorConfig:
    """加载监控配置；HOST/PORT/API_KEY 未在 monitor.env 显式设置时，
    自动继承 server.env 的推理服务地址——避免两处配置不一致导致
    采样器探测错误端口。"""
    values = _read_monitor_values(path)
    explicit = set(values)
    known = {k: values[k] for k in MONITOR_KEYS if k in values}
    if server_env_path is not None and server_env_path.exists():
        from .config import ServerConfig, parse_env_file

        svals, _ = parse_env_file(server_env_path, "server")
        scfg = ServerConfig(**svals)
        for key in ("HOST", "PORT", "API_KEY"):
            if key not in explicit:
                known[key] = getattr(scfg, key)
    known = {k: known[k] for k in MONITOR_KEYS if k in known}
    cfg = MonitorConfig(**known)
    errors = cfg.validate()
    if errors:
        raise ValueError("monitor.env 配置无效：" + "；".join(errors))
    return cfg


def save_monitor_config(cfg: MonitorConfig, path: Path) -> None:
    lines = [
        "# llamacpp monitor configuration; parsed as data, never executed.",
        f"INTERVAL={cfg.INTERVAL}",
        f"TEMP_MAX={cfg.TEMP_MAX}",
        f"MEM_PCT_MAX={cfg.MEM_PCT_MAX}",
        f"HEALTH_FAIL_MAX={cfg.HEALTH_FAIL_MAX}",
        f"ALERT_COOLDOWN={cfg.ALERT_COOLDOWN}",
        f"WEBHOOK_URL={cfg.WEBHOOK_URL}",
        f"WEBHOOK_FORMAT={cfg.WEBHOOK_FORMAT}",
        f"TELEGRAM_CHAT_ID={cfg.TELEGRAM_CHAT_ID}",
        f"API_KEY={cfg.API_KEY}",
        f"HOST={cfg.HOST}",
        f"PORT={cfg.PORT}",
        f"PANEL_KEY={cfg.PANEL_KEY}",
    ]
    atomic_write(path, "\n".join(lines) + "\n")


# --------------------------------------------------------------- 数据库 ----

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    gpu_index INTEGER NOT NULL,
    name TEXT,
    mem_used_mib INTEGER,
    mem_total_mib INTEGER,
    mem_pct INTEGER,
    temperature INTEGER,
    utilization INTEGER,
    predicted_tokens INTEGER,
    prompt_tokens INTEGER,
    predicted_tps REAL
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    rule TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


# -------------------------------------------------------------- 采集器 ----


@dataclass(frozen=True)
class GpuSample:
    index: int
    name: str
    mem_used_mib: int
    mem_total_mib: int
    temperature: int | None
    utilization: int | None


def collect_gpu_samples() -> list[GpuSample]:
    """nvidia-smi 查询；不可用时返回空列表。"""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []

    def to_int(value: str) -> int | None:
        digits = "".join(ch for ch in value.strip() if ch.isdigit())
        return int(digits) if digits else None

    samples = []
    for line in proc.stdout.splitlines():
        cells = [c.strip() for c in line.split(",") if c.strip()]
        if len(cells) < 4:
            continue
        samples.append(
            GpuSample(
                index=int(cells[0]) if cells[0].isdigit() else 0,
                name=cells[1],
                mem_used_mib=int(cells[2]) if cells[2].isdigit() else 0,
                mem_total_mib=int(cells[3]) if cells[3].isdigit() else 0,
                temperature=to_int(cells[4]) if len(cells) > 4 else None,
                utilization=to_int(cells[5]) if len(cells) > 5 else None,
            )
        )
    return samples


METRIC_PATTERNS = {
    "prompt_tokens": re.compile(r"^llamacpp:prompt_tokens_total\s+(\S+)"),
    "predicted_tokens": re.compile(r"^llamacpp:tokens_predicted_total\s+(\S+)"),
}


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """从 llama-server /metrics 的 Prometheus 文本提取 token 计数。"""
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        for key, pattern in METRIC_PATTERNS.items():
            match = pattern.match(line)
            if match:
                try:
                    result[key] = float(match.group(1))
                except ValueError:
                    pass
    return result


def collect_server_metrics(host: str, port: str, api_key: str = "") -> dict[str, float] | None:
    """拉取 /metrics；服务不可达时返回 None。"""
    from .server import normalize_host

    url = f"http://{normalize_host(host)}:{port}/metrics"
    try:
        code, body = http_get(url, api_key or None, timeout=5)
    except Exception:  # noqa: BLE001 — 网络层任何失败都视为不可达
        return None
    if code != 200:
        return None
    return parse_prometheus_metrics(body.decode(errors="replace"))


def check_health(host: str, port: str) -> bool:
    try:
        code, _ = http_get(health_url(host, port), timeout=5)
    except Exception:  # noqa: BLE001
        return False
    return code == 200


# -------------------------------------------------------------- 告警 ------


def webhook_payload(fmt: str, title: str, message: str, chat_id: str = "") -> tuple[str, bytes]:
    """按格式构造 (content_type, body)。URL 由调用方决定。"""
    if fmt == "wecom":
        payload = {"msgtype": "text", "text": {"content": f"[{title}] {message}"}}
    elif fmt == "dingtalk":
        payload = {"msgtype": "text", "text": {"content": f"[{title}] {message}"}}
    elif fmt == "telegram":
        payload = {"chat_id": chat_id, "text": f"[{title}] {message}"}
    elif fmt == "bark":
        payload = {"title": title, "body": message}
    else:
        payload = {"title": title, "message": message, "level": "warning"}
    return "application/json", json.dumps(payload, ensure_ascii=False).encode()


def send_webhook(url: str, fmt: str, title: str, message: str, chat_id: str = "") -> bool:
    content_type, body = webhook_payload(fmt, title, message, chat_id)
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 — 通知失败不应中断采样循环
        return False


class Alerter:
    """带同类告警冷却的告警器。"""

    def __init__(self, cfg: MonitorConfig, db: sqlite3.Connection) -> None:
        self.cfg = cfg
        self.db = db
        self.last_fired: dict[str, float] = {}
        self.health_fail_streak = 0

    def fire(self, rule: str, message: str, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        last = self.last_fired.get(rule, 0.0)
        if now - last < self.cfg.ALERT_COOLDOWN:
            return False
        delivered = False
        if self.cfg.WEBHOOK_URL:
            delivered = send_webhook(
                self.cfg.WEBHOOK_URL, self.cfg.WEBHOOK_FORMAT,
                "llamacpp 监控告警", message, self.cfg.TELEGRAM_CHAT_ID,
            )
        self.db.execute(
            "INSERT INTO alerts (ts, rule, level, message, delivered) VALUES (?,?,?,?,?)",
            (now, rule, "warning", message, int(delivered)),
        )
        self.db.commit()
        self.last_fired[rule] = now
        print(f"[ALERT] {rule}: {message}", flush=True)
        return True

    def evaluate(self, samples: list[GpuSample], server_up: bool, now: float | None = None) -> list[str]:
        fired: list[str] = []
        for sample in samples:
            if sample.temperature is not None and sample.temperature >= self.cfg.TEMP_MAX:
                if self.fire(
                    f"gpu{sample.index}.temperature",
                    f"GPU{sample.index} 温度 {sample.temperature}℃ 超过阈值 {self.cfg.TEMP_MAX}℃",
                    now,
                ):
                    fired.append(f"gpu{sample.index}.temperature")
            total = sample.mem_total_mib
            if total > 0:
                pct = round(sample.mem_used_mib * 100 / total)
                if pct >= self.cfg.MEM_PCT_MAX:
                    if self.fire(
                        f"gpu{sample.index}.memory",
                        f"GPU{sample.index} 显存使用率 {pct}% 超过阈值 {self.cfg.MEM_PCT_MAX}%",
                        now,
                    ):
                        fired.append(f"gpu{sample.index}.memory")
        if server_up:
            self.health_fail_streak = 0
        else:
            self.health_fail_streak += 1
            if self.health_fail_streak == self.cfg.HEALTH_FAIL_MAX:
                if self.fire(
                    "health",
                    f"llama-server health 连续 {self.health_fail_streak} 次检查失败",
                    now,
                ):
                    fired.append("health")
        return fired


# ------------------------------------------------------------ 采样循环 ----


def sample_once(db: sqlite3.Connection, cfg: MonitorConfig, alerter: Alerter,
                prev_predicted: float | None, diag: dict | None = None
                ) -> tuple[float | None, list[str]]:
    """执行一轮采集、入库与告警评估，返回 (最新 predicted_tokens, 触发的告警)。

    diag 传入 dict 时回填诊断信息：health / metrics / counters。
    """
    ts = time.time()
    gpu_samples = collect_gpu_samples()
    metrics = collect_server_metrics(cfg.HOST, cfg.PORT, cfg.API_KEY)
    server_up = check_health(cfg.HOST, cfg.PORT)
    if diag is not None:
        diag["health"] = server_up
        diag["metrics"] = bool(metrics and "predicted_tokens" in metrics)

    predicted = metrics.get("predicted_tokens") if metrics else None
    tps = None
    if predicted is not None and prev_predicted is not None and cfg.INTERVAL > 0:
        tps = max((predicted - prev_predicted) / cfg.INTERVAL, 0.0)

    prompt_tokens = (metrics or {}).get("prompt_tokens")
    rows = []
    for s in gpu_samples:
        pct = round(s.mem_used_mib * 100 / s.mem_total_mib) if s.mem_total_mib > 0 else None
        rows.append((
            ts, s.index, s.name, s.mem_used_mib, s.mem_total_mib,
            pct, s.temperature, s.utilization,
            predicted, prompt_tokens,
            tps,
        ))
    # 服务端指标独立成行（gpu_index=-1）：即使无 GPU 数据也要保证
    # 吞吐序列入库，否则面板永远看不到 tok/s。
    if gpu_samples == [] and (predicted is not None or tps is not None):
        rows.append((ts, -1, None, None, None, None, None, None,
                     predicted, prompt_tokens, tps))
    if rows:
        db.executemany(
            "INSERT INTO samples (ts, gpu_index, name, mem_used_mib, mem_total_mib,"
            " mem_pct, temperature, utilization, predicted_tokens, prompt_tokens,"
            " predicted_tps) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        db.commit()

    fired = alerter.evaluate(gpu_samples, server_up)
    return predicted, fired


def run_loop(db_path: Path, cfg: MonitorConfig, once: bool = False) -> None:
    db = connect(db_path)
    alerter = Alerter(cfg, db)
    print(f"[INFO] 监控启动：间隔 {cfg.INTERVAL}s，数据写入 {db_path}")
    prev: float | None = None
    while True:
        try:
            prev, _fired = sample_once(db, cfg, alerter, prev)
        except Exception as exc:  # noqa: BLE001 — 单轮失败不退出
            print(f"[WARN] 采样异常：{exc}", flush=True)
        if once:
            return
        time.sleep(cfg.INTERVAL)


# --------------------------------------------------------------- 查询 ----


def latest_tps(db: sqlite3.Connection, limit: int = 60) -> list[tuple[float, float]]:
    """按采样轮（时间戳）返回吞吐序列；同轮多卡只取一条。"""
    rows = db.execute(
        "SELECT ts, MAX(predicted_tps) FROM samples WHERE predicted_tps IS NOT NULL "
        "GROUP BY ts ORDER BY ts DESC LIMIT ?", (limit,),
    ).fetchall()
    return [(float(ts), float(tps)) for ts, tps in reversed(rows)]


def recent_alerts(db: sqlite3.Connection, limit: int = 50) -> list[tuple]:
    return db.execute(
        "SELECT ts, rule, level, message, delivered FROM alerts ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
