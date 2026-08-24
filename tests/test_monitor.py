"""监控模块测试：指标解析、采样入库、告警阈值与 Webhook 负载。"""

from __future__ import annotations

import json

import pytest

from llamacpp.monitor import (
    Alerter,
    GpuSample,
    MonitorConfig,
    connect,
    latest_tps,
    load_monitor_config,
    parse_prometheus_metrics,
    recent_alerts,
    sample_once,
    save_monitor_config,
    webhook_payload,
)


METRICS_TEXT = """\
# HELP llamacpp:tokens_predicted_total Number of predicted tokens.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:prompt_tokens_total 1234
llamacpp:tokens_predicted_total 5678
# TYPE other_metric gauge
other_metric 99
"""


class TestParseMetrics:
    def test_extract_counters(self):
        result = parse_prometheus_metrics(METRICS_TEXT)
        assert result == {"prompt_tokens": 1234.0, "predicted_tokens": 5678.0}

    def test_garbage_safe(self):
        assert parse_prometheus_metrics("not a metric") == {}
        assert parse_prometheus_metrics("llamacpp:tokens_predicted_total abc") == {}


class TestMonitorConfig:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "monitor.env"
        save_monitor_config(MonitorConfig(TEMP_MAX=90, WEBHOOK_URL="https://x"), path)
        cfg = load_monitor_config(path)
        assert cfg.TEMP_MAX == 90
        assert cfg.WEBHOOK_URL == "https://x"
        assert cfg.INTERVAL == 3   # 默认值保留

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "monitor.env"
        path.write_text("BOGUS=1\nTEMP_MAX=88\n", encoding="utf-8")
        assert load_monitor_config(path).TEMP_MAX == 88

    def test_invalid_format_raises(self, tmp_path):
        path = tmp_path / "monitor.env"
        path.write_text("WEBHOOK_FORMAT=email\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_monitor_config(path)


class TestWebhookPayload:
    FORMATS = {
        "generic": {"title", "message", "level"},
        "wecom": {"msgtype", "text"},
        "dingtalk": {"msgtype", "text"},
        "telegram": {"chat_id", "text"},
        "bark": {"title", "body"},
    }

    @pytest.mark.parametrize("fmt", list(FORMATS))
    def test_each_format(self, fmt):
        content_type, body = webhook_payload(fmt, "标题", "内容", chat_id="42")
        assert content_type == "application/json"
        data = json.loads(body)
        assert set(data) == self.FORMATS[fmt]
        if fmt == "telegram":
            assert data["chat_id"] == "42"


def make_samples(temp=60, used_mib=8000, total_mib=12000):
    return [
        GpuSample(index=0, name="RTX", mem_used_mib=used_mib,
                  mem_total_mib=total_mib, temperature=temp, utilization=50),
        GpuSample(index=1, name="RTX", mem_used_mib=used_mib,
                  mem_total_mib=total_mib, temperature=temp - 5, utilization=30),
    ]


class TestAlerter:
    def make_alerter(self, tmp_path, **cfg_overrides):
        db = connect(tmp_path / "m.db")
        cfg = MonitorConfig(ALERT_COOLDOWN=0, **{
            k: v for k, v in cfg_overrides.items() if k != "ALERT_COOLDOWN"
        })
        if cfg_overrides.get("ALERT_COOLDOWN") is not None:
            cfg.ALERT_COOLDOWN = cfg_overrides["ALERT_COOLDOWN"]
        return Alerter(cfg, db), db

    def test_temperature_alert(self, tmp_path):
        alerter, db = self.make_alerter(tmp_path, TEMP_MAX=85)
        fired = alerter.evaluate(make_samples(temp=90), server_up=True)
        assert "gpu0.temperature" in fired
        rows = recent_alerts(db)
        assert len(rows) == 2  # gpu0 和 gpu1 都超温（90 和 85）
        assert "温度" in rows[0][3]

    def test_memory_pct_alert(self, tmp_path):
        alerter, db = self.make_alerter(tmp_path, MEM_PCT_MAX=95)
        fired = alerter.evaluate(make_samples(used_mib=11800), server_up=True)
        assert fired  # 98% > 95%

    def test_health_streak_alert_exactly_at_threshold(self, tmp_path):
        alerter, _db = self.make_alerter(tmp_path, HEALTH_FAIL_MAX=3)
        assert alerter.evaluate([], server_up=False) == []
        assert alerter.evaluate([], server_up=False) == []
        fired = alerter.evaluate([], server_up=False)
        assert fired == ["health"]
        # 持续失败不再重复告警（streak 已超过阈值）
        assert alerter.evaluate([], server_up=False) == []

    def test_cooldown_suppresses_repeat(self, tmp_path):
        alerter, db = self.make_alerter(tmp_path, ALERT_COOLDOWN=600)
        now = 1000.0
        assert alerter.evaluate(make_samples(temp=90), True, now=now)
        assert not alerter.evaluate(make_samples(temp=90), True, now=now + 100)
        # 冷却期后再次触发，但 evaluate 返回值取决于 last_fired 更新
        assert alerter.evaluate(make_samples(temp=95), True, now=now + 700)


class TestSampleOnce:
    def test_insert_and_tps(self, tmp_path, monkeypatch):
        import llamacpp.monitor as mon

        monkeypatch.setattr(mon, "collect_gpu_samples",
                            lambda: make_samples(temp=60, used_mib=4000))
        metrics_sequence = iter([
            {"predicted_tokens": 1000.0, "prompt_tokens": 10.0},
            {"predicted_tokens": 1050.0, "prompt_tokens": 20.0},
        ])
        monkeypatch.setattr(mon, "collect_server_metrics",
                            lambda *a, **k: next(metrics_sequence))
        monkeypatch.setattr(mon, "check_health", lambda *a, **k: True)

        db = connect(tmp_path / "m.db")
        cfg = MonitorConfig(INTERVAL=10)
        alerter = Alerter(cfg, db)

        prev, fired = sample_once(db, cfg, alerter, None)
        assert prev == 1000.0 and fired == []
        prev, fired = sample_once(db, cfg, alerter, prev)
        assert fired == []
        tps_points = latest_tps(db)
        assert len(tps_points) == 1
        assert abs(tps_points[0][1] - 5.0) < 1e-6  # 50 tokens / 10s

    def test_gpu_down_no_crash(self, tmp_path, monkeypatch):
        import llamacpp.monitor as mon

        monkeypatch.setattr(mon, "collect_gpu_samples", lambda: [])
        monkeypatch.setattr(mon, "collect_server_metrics", lambda *a, **k: None)
        monkeypatch.setattr(mon, "check_health", lambda *a, **k: False)
        db = connect(tmp_path / "m.db")
        cfg = MonitorConfig(HEALTH_FAIL_MAX=1)
        alerter = Alerter(cfg, db)
        prev, fired = sample_once(db, cfg, alerter, None)
        assert prev is None
        assert fired == ["health"]


class TestQuotedValues:
    """回归：用户常按 shell 习惯写 KEY="xxx"，解析必须去引号。"""

    def test_double_quotes_stripped(self, tmp_path):
        path = tmp_path / "monitor.env"
        path.write_text('PANEL_KEY="my-secret"\n', encoding="utf-8")
        assert load_monitor_config(path).PANEL_KEY == "my-secret"

    def test_single_quotes_stripped(self, tmp_path):
        path = tmp_path / "monitor.env"
        path.write_text("PANEL_KEY='my-secret'\n", encoding="utf-8")
        assert load_monitor_config(path).PANEL_KEY == "my-secret"

    def test_plain_value_untouched(self, tmp_path):
        path = tmp_path / "monitor.env"
        path.write_text("PANEL_KEY=my-secret\n", encoding="utf-8")
        assert load_monitor_config(path).PANEL_KEY == "my-secret"

    def test_unpaired_quote_kept(self, tmp_path):
        path = tmp_path / "monitor.env"
        path.write_text('PANEL_KEY=my"secret\n', encoding="utf-8")
        assert load_monitor_config(path).PANEL_KEY == 'my"secret'


class TestConfigInheritance:
    """回归：monitor.env 未显式设置 HOST/PORT/API_KEY 时必须继承 server.env，
    否则采样器探测默认 8080 而推理服务跑在其他端口，导致吞吐永远为空。"""

    def test_inherits_from_server_env(self, tmp_path):
        from llamacpp.monitor import load_monitor_config

        server_env = tmp_path / "server.env"
        server_env.write_text("PORT=8123\nHOST=0.0.0.0\nAPI_KEY=sk-x\n",
                              encoding="utf-8")
        monitor = tmp_path / "monitor.env"
        monitor.write_text("INTERVAL=5\n", encoding="utf-8")
        cfg = load_monitor_config(monitor, server_env_path=server_env)
        assert cfg.PORT == "8123"
        assert cfg.API_KEY == "sk-x"

    def test_explicit_monitor_values_win(self, tmp_path):
        from llamacpp.monitor import load_monitor_config

        server_env = tmp_path / "server.env"
        server_env.write_text("PORT=8123\n", encoding="utf-8")
        monitor = tmp_path / "monitor.env"
        monitor.write_text("PORT=9999\nTEMP_MAX=90\n", encoding="utf-8")
        cfg = load_monitor_config(monitor, server_env_path=server_env)
        assert cfg.PORT == "9999"   # 显式设置优先
        assert cfg.TEMP_MAX == 90

    def test_missing_server_env_uses_defaults(self, tmp_path):
        from llamacpp.monitor import load_monitor_config

        cfg = load_monitor_config(tmp_path / "monitor.env",
                                  server_env_path=tmp_path / "nope.env")
        assert cfg.PORT == "8080"


class TestServerMetricsWithoutGpu:
    """回归：无 nvidia-smi 数据时，服务端吞吐也必须入库，
    否则面板吞吐永远为空。"""

    def test_tps_persisted_without_gpu_samples(self, tmp_path, monkeypatch):
        import llamacpp.monitor as mon

        monkeypatch.setattr(mon, "collect_gpu_samples", lambda: [])
        seq = iter([{"predicted_tokens": 100.0},
                    {"predicted_tokens": 150.0}])
        monkeypatch.setattr(mon, "collect_server_metrics",
                            lambda *a, **k: next(seq))
        monkeypatch.setattr(mon, "check_health", lambda *a, **k: True)

        db = connect(tmp_path / "m.db")
        cfg = MonitorConfig(INTERVAL=10)
        alerter = Alerter(cfg, db)
        prev, _ = sample_once(db, cfg, alerter, None)
        prev, _ = sample_once(db, cfg, alerter, prev)
        points = latest_tps(db)
        assert len(points) == 1
        assert abs(points[0][1] - 5.0) < 1e-6   # 50 tokens / 10s

    def test_no_metrics_no_rows(self, tmp_path, monkeypatch):
        import llamacpp.monitor as mon

        monkeypatch.setattr(mon, "collect_gpu_samples", lambda: [])
        monkeypatch.setattr(mon, "collect_server_metrics", lambda *a, **k: None)
        monkeypatch.setattr(mon, "check_health", lambda *a, **k: True)
        db = connect(tmp_path / "m.db")
        sample_once(db, MonitorConfig(), Alerter(MonitorConfig(), db), None)
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0
