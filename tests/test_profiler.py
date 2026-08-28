"""profiler.py — sampler maths and the FastAPI endpoints.

These run without mlx (it's Apple-silicon only): profiler.py imports with
mx=None, _stat() then returns None and the MLX gauges read 0. Everything
else — psutil, the counters, the app — works anywhere.
"""
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import profiler


def cval(name):
    return REGISTRY.get_sample_value(name) or 0.0


class FakeThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


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


# --- endpoints ---------------------------------------------------------

def test_snapshot_is_503_before_the_first_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    client = TestClient(profiler.build_app(sampler, FakeThread()))
    assert client.get("/api/snapshot").status_code == 503


def test_snapshot_and_metrics_after_a_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeThread()))

    body = client.get("/api/snapshot").json()
    assert body["system"]["total_bytes"] > 0
    assert set(body) >= {"timestamp", "mlx", "process", "system", "macos"}

    text = client.get("/metrics").text
    assert "macos_system_memory_used_bytes" in text
    assert "mlx_process_memory_bytes" in text


def test_healthz_503_without_a_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    client = TestClient(profiler.build_app(sampler, FakeThread()))
    assert client.get("/healthz").status_code == 503


def test_healthz_200_after_a_recent_sample():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeThread()))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["healthy"] is True


def test_healthz_503_when_model_server_thread_is_dead():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeThread(alive=False)))
    assert client.get("/healthz").status_code == 503


# --- bearer-token auth -------------------------------------------------

def test_no_token_leaves_endpoints_open():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeThread()))
    assert client.get("/metrics").status_code == 200
    assert client.get("/api/snapshot").status_code == 200


def test_token_required_on_metrics_and_snapshot():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeThread(), token="s3cret"))

    assert client.get("/metrics").status_code == 401
    assert client.get("/api/snapshot").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401

    ok = {"Authorization": "Bearer s3cret"}
    assert client.get("/metrics", headers=ok).status_code == 200
    assert client.get("/api/snapshot", headers=ok).status_code == 200


def test_healthz_stays_open_with_a_token_set():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    sampler.sample_once()
    client = TestClient(profiler.build_app(sampler, FakeThread(), token="s3cret"))
    assert client.get("/healthz").status_code == 200


def test_sample_once_records_a_timestamp():
    sampler = profiler.Sampler(interval=1, use_footprint=False)
    assert sampler.last_sample_ts == 0.0
    sampler.sample_once()
    assert sampler.last_sample_ts > 0.0
