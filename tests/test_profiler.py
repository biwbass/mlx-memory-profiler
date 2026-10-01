"""profiler.py — sampler maths, MLX-memory scrape, and the FastAPI endpoints.

These run anywhere: profiler.py no longer imports mlx (the model server is a
child process passed after `--`), so psutil, the counters, the HTTP scrape
parsing and the app are all exercised off Apple Silicon too.
"""
import urllib.error

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import profiler


def cval(name):
    return REGISTRY.get_sample_value(name) or 0.0


class FakeProc:
    """Stand-in for the supervised model-server subprocess."""

    def __init__(self, alive=True):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


# --- _counter_delta ------------------------------------------------------

def test_counter_delta_first_reading_contributes_nothing():
    assert profiler._counter_delta(None, 5000) == 0


def test_counter_delta_normal_increase():
    assert profiler._counter_delta(100, 175) == 75


def test_counter_delta_flat():
    assert profiler._counter_delta(50, 50) == 0


def test_counter_delta_reset_uses_new_value():
    assert profiler._counter_delta(1_000_000, 42) == 42


def test_bump_counters_baseline_then_delta():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    start = cval("macos_vm_pageins_pages_total")

    sampler._bump_counters({"pageins": 1000})           # adopt baseline
    assert cval("macos_vm_pageins_pages_total") == start

    sampler._bump_counters({"pageins": 1120})           # +120
    assert cval("macos_vm_pageins_pages_total") == start + 120


# --- scrape_mlx_memory -------------------------------------------------

def _fake_urlopen(body):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return body if isinstance(body, bytes) else body.encode()

    return lambda url, timeout=None: _Resp()


def test_scrape_mlx_memory_flat_keys(monkeypatch):
    monkeypatch.setattr(
        profiler.urllib.request, "urlopen",
        _fake_urlopen('{"active_bytes": 10, "cache_bytes": 4, "peak_bytes": 12}'),
    )
    assert profiler.scrape_mlx_memory("http://x/metrics") == {"active": 10, "cache": 4, "peak": 12}


def test_scrape_mlx_memory_nested_and_partial(monkeypatch):
    monkeypatch.setattr(
        profiler.urllib.request, "urlopen",
        _fake_urlopen('{"memory": {"active": 7}}'),
    )
    assert profiler.scrape_mlx_memory("http://x") == {"active": 7}


def test_scrape_mlx_memory_swallows_errors(monkeypatch):
    def boom(url, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(profiler.urllib.request, "urlopen", boom)
    assert profiler.scrape_mlx_memory("http://x") is None


def test_scrape_mlx_memory_bad_body(monkeypatch):
    monkeypatch.setattr(profiler.urllib.request, "urlopen", _fake_urlopen("not json"))
    assert profiler.scrape_mlx_memory("http://x") is None


# --- _derive_server_url ----------------------------------------------

def test_derive_server_url_explicit_wins():
    assert profiler._derive_server_url("http://h:9/", ["x", "--port", "1"]) == "http://h:9"


def test_derive_server_url_from_cmd_flags():
    assert profiler._derive_server_url(None, ["mlx-openai-server", "launch", "--port", "8080"]) == \
        "http://127.0.0.1:8080"


def test_derive_server_url_equals_form_and_wildcard_host():
    got = profiler._derive_server_url(None, ["s", "--host=0.0.0.0", "--port=1234"])
    assert got == "http://127.0.0.1:1234"


def test_derive_server_url_default():
    assert profiler._derive_server_url(None, ["mlx-openai-server", "launch"]) == "http://127.0.0.1:8000"


# --- Sampler MLX-memory wiring --------------------------------------

def test_sample_without_mlx_memory_url_marks_unavailable():
    sampler = profiler.Sampler(interval=1, use_footprint=False)   # no url
    sampler.sample_once()
    assert sampler.latest["mlx"] == {"available": False}


def test_sample_scrapes_and_sets_gauges(monkeypatch):
    monkeypatch.setattr(
        profiler, "scrape_mlx_memory",
        lambda url, timeout=1.5: {"active": 900, "cache": 100, "peak": 950},
    )
    sampler = profiler.Sampler(interval=1, use_footprint=False, mlx_memory_url="http://model/x")
    sampler.sample_once()

    assert cval("mlx_active_memory_bytes") == 900
    assert cval("mlx_peak_memory_bytes") == 950
    assert cval("mlx_profiler_mlx_memory_up") == 1
    assert sampler.latest["mlx"]["available"] is True
    assert sampler.latest["mlx"]["active_bytes"] == 900


def test_sample_counts_scrape_failures(monkeypatch):
    monkeypatch.setattr(profiler, "scrape_mlx_memory", lambda url, timeout=1.5: None)
    sampler = profiler.Sampler(interval=1, use_footprint=False, mlx_memory_url="http://model/x")
    start = cval("mlx_profiler_mlx_memory_scrape_errors_total")
    sampler.sample_once()
    assert cval("mlx_profiler_mlx_memory_scrape_errors_total") == start + 1
    assert cval("mlx_profiler_mlx_memory_up") == 0
    assert sampler.latest["mlx"] == {"available": False}


# --- endpoints ---------------------------------------------------------

def test_snapshot_is_503_before_the_first_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    client = TestClient(profiler.build_app(sampler, FakeProc()))
    assert client.get("/api/snapshot").status_code == 503


def test_snapshot_and_metrics_after_a_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeProc()))

    body = client.get("/api/snapshot").json()
    assert body["system"]["total_bytes"] > 0
    assert set(body) >= {"timestamp", "mlx", "process", "system", "macos"}

    text = client.get("/metrics").text
    assert "macos_system_memory_used_bytes" in text
    assert "mlx_process_memory_bytes" in text


def test_healthz_503_without_a_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    client = TestClient(profiler.build_app(sampler, FakeProc()))
    assert client.get("/healthz").status_code == 503


def test_healthz_200_after_a_recent_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeProc()))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["healthy"] is True


def test_healthz_503_when_model_server_is_dead():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeProc(alive=False)))
    assert client.get("/healthz").status_code == 503


def test_sample_once_records_a_timestamp():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    assert sampler.last_sample_ts == 0.0
    sampler.sample_once()
    assert sampler.last_sample_ts > 0.0


# --- bearer-token auth -------------------------------------------------

def test_no_token_leaves_endpoints_open():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeProc()))
    assert client.get("/metrics").status_code == 200
    assert client.get("/api/snapshot").status_code == 200


def test_token_required_on_metrics_and_snapshot():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeProc(), token="s3cret"))

    assert client.get("/metrics").status_code == 401
    assert client.get("/api/snapshot").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401

    ok = {"Authorization": "Bearer s3cret"}
    assert client.get("/metrics", headers=ok).status_code == 200
    assert client.get("/api/snapshot", headers=ok).status_code == 200


def test_healthz_stays_open_with_a_token_set():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeProc(), token="s3cret"))
    assert client.get("/healthz").status_code == 200
