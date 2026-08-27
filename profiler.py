#!/usr/bin/env python3
"""
Run `mlx_lm.server` with a Prometheus metrics endpoint bolted on.

This is a drop-in wrapper: pass `mlx_lm.server`'s own flags straight through
and it serves the model exactly as before, on the same port. Alongside it,
this process also runs a small FastAPI app that exposes `/metrics` in
Prometheus exposition format, so a Prometheus server (e.g. on a k8s cluster
that can reach this Mac over the LAN) can scrape live memory usage and a
Grafana dashboard can graph it.

Like serve_kv.py, `mlx_lm.server` is loaded as a module and run in a
background thread of *this* process, so the sampler can read MLX's own
allocator counters (active / cache / peak memory) directly. It also samples
the process-tree footprint (procmem.py) and the full macOS memory picture —
Activity Monitor breakdown, memory pressure, paging counters (macstat.py).

Usage:

    python profiler.py --model mlx-community/Qwen3.5-9B-6bit --port 8080

    # model server -> :8080 (unchanged, OpenAI-compatible)
    # metrics      -> :9105/metrics

    # tune the sampler / move the metrics port:
    python profiler.py --model ... --metrics-port 9105 --sample-interval 2

Requires: pip install psutil mlx-lm fastapi "uvicorn[standard]" prometheus-client

Caveat (inherited from serve_kv.py): relies on `mlx_lm.server.main()` reading
sys.argv, which holds for the mlx-lm versions this targets but isn't a
guaranteed public API.
"""
import argparse
import atexit
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    sys.exit("This script requires psutil: pip install psutil")

try:
    import mlx.core as mx
except ImportError:
    sys.exit("This script requires mlx: pip install mlx-lm")

try:
    from mlx_lm import server as mlx_server
except ImportError:
    sys.exit("This script requires mlx-lm: pip install mlx-lm")

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
except ImportError:
    sys.exit(
        'This script requires the service extras: '
        'pip install fastapi "uvicorn[standard]" prometheus-client'
    )

from macstat import mac_memory_stats
from mlxinfo import device_memory_info
from procmem import HAVE_FOOTPRINT, tree_footprint_bytes, tree_procs, tree_rss_bytes

GB = 1024 ** 3

# --- metrics ---------------------------------------------------------------

# MLX allocator counters (weights + KV cache together; see serve_kv.py notes).
_mlx_active = Gauge("mlx_active_memory_bytes", "MLX active memory: model weights + KV cache")
_mlx_cache = Gauge("mlx_cache_memory_bytes", "MLX allocator cache (freed buffers held for reuse)")
_mlx_peak = Gauge("mlx_peak_memory_bytes", "MLX peak active memory since process start")

# This process' tree, via `footprint` (phys_footprint) when available, else RSS.
_proc_mem = Gauge("mlx_process_memory_bytes", "Profiler process-tree memory", ["source"])

# Static ceilings from mx.device_info() (see mlxinfo.py).
_gpu_total = Gauge("mlx_gpu_total_memory_bytes", "Total unified memory on this machine")
_gpu_working_set = Gauge(
    "mlx_gpu_recommended_working_set_bytes",
    "Apple's recommended GPU working-set ceiling (mlx_lm.server's wired limit)",
)

# System memory, psutil headline numbers.
_sys_total = Gauge("macos_system_memory_total_bytes", "Total physical memory")
_sys_used = Gauge("macos_system_memory_used_bytes", "System memory in use (psutil)")
_sys_used_pct = Gauge("macos_system_memory_used_percent", "System memory used, percent (psutil)")
_swap_used = Gauge("macos_swap_used_bytes", "Swap in use")
_swap_total = Gauge("macos_swap_total_bytes", "Swap configured")

# macOS Activity Monitor breakdown (macstat.py, from vm_stat).
_mac_wired = Gauge("macos_memory_wired_bytes", "Wired (non-pageable) memory")
_mac_compressed = Gauge("macos_memory_compressed_bytes", "Memory held by the compressor")
_mac_app = Gauge("macos_memory_app_bytes", "App memory (anonymous minus purgeable)")
_mac_anonymous = Gauge("macos_memory_anonymous_bytes", "Anonymous pages")
_mac_purgeable = Gauge("macos_memory_purgeable_bytes", "Purgeable pages")
_mac_cached_files = Gauge("macos_memory_cached_files_bytes", "File-backed (cached files) pages")
_mac_free = Gauge("macos_memory_free_bytes", "Free pages")
_mac_used = Gauge("macos_memory_used_bytes", "Activity Monitor 'Memory Used' (wired + compressed + app)")
_mac_pressure = Gauge("macos_memory_pressure_level", "Memory pressure: 1 normal, 2 warning, 4 critical")

# macOS paging activity — cumulative kernel counters, exported as counters.
_PAGE_COUNTERS = {
    "pageins": Counter("macos_vm_pageins_pages_total", "Pages paged in from disk"),
    "pageouts": Counter("macos_vm_pageouts_pages_total", "Pages paged out to disk"),
    "swapins": Counter("macos_vm_swapins_pages_total", "Pages swapped in"),
    "swapouts": Counter("macos_vm_swapouts_pages_total", "Pages swapped out"),
    "compressions": Counter("macos_vm_compressions_pages_total", "Pages compressed"),
    "decompressions": Counter("macos_vm_decompressions_pages_total", "Pages decompressed"),
    "faults": Counter("macos_vm_faults_total", "Translation faults"),
}

# Health / liveness.
_up = Gauge("mlx_profiler_up", "1 while the profiler process is running")
_server_up = Gauge("mlx_profiler_model_server_up", "1 while the mlx_lm.server thread is alive")
_last_sample = Gauge("mlx_profiler_last_sample_timestamp_seconds", "Unix time of the last successful sample")
_sample_errors = Counter("mlx_profiler_sample_errors_total", "Sampler iterations that raised")


def _stat(*names):
    for name in names:
        fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
        if fn is not None:
            return fn()
    return None


class Sampler:
    """Background loop: read every source once per interval, update metrics,
    and keep the latest reading around for the /api/snapshot endpoint."""

    def __init__(self, interval, use_footprint):
        self.interval = interval
        self.use_footprint = use_footprint
        self.proc = psutil.Process()
        self.latest = {}
        self.last_sample_ts = 0.0
        self._page_prev = {}
        self._stop = threading.Event()
        self._fp_tmp = None
        if use_footprint:
            fd, self._fp_tmp = tempfile.mkstemp(suffix=".json", prefix="mlxprof_")
            os.close(fd)
            atexit.register(self._cleanup)

    def _cleanup(self):
        if self._fp_tmp:
            try:
                os.unlink(self._fp_tmp)
            except OSError:
                pass

    def _bump_counters(self, mac):
        for key, counter in _PAGE_COUNTERS.items():
            value = mac.get(key)
            if value is None:
                continue
            prev = self._page_prev.get(key)
            if prev is None:
                delta = 0  # first reading: adopt the baseline, don't emit the lifetime total
            elif value >= prev:
                delta = value - prev
            else:
                delta = value  # counter reset (reboot)
            if delta:
                counter.inc(delta)
            self._page_prev[key] = value

    def sample_once(self):
        active = _stat("get_active_memory") or 0
        cache = _stat("get_cache_memory") or 0
        peak = _stat("get_peak_memory") or 0
        _mlx_active.set(active)
        _mlx_cache.set(cache)
        _mlx_peak.set(peak)

        procs = tree_procs(self.proc)
        proc_bytes = tree_footprint_bytes(procs, self._fp_tmp) if self.use_footprint else None
        source = "footprint"
        if proc_bytes is None:
            proc_bytes = tree_rss_bytes(procs)
            source = "rss"
        _proc_mem.labels(source=source).set(proc_bytes)

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        _sys_total.set(vm.total)
        _sys_used.set(vm.used)
        _sys_used_pct.set(vm.percent)
        _swap_used.set(swap.used)
        _swap_total.set(swap.total)

        mac = mac_memory_stats()
        if mac:
            _mac_wired.set(mac["wired_bytes"])
            _mac_compressed.set(mac["compressed_bytes"])
            _mac_app.set(mac["app_bytes"])
            _mac_anonymous.set(mac["anonymous_bytes"])
            _mac_purgeable.set(mac["purgeable_bytes"])
            _mac_cached_files.set(mac["cached_files_bytes"])
            _mac_free.set(mac["free_bytes"])
            _mac_used.set(mac["used_bytes"])
            if mac["pressure_level"] is not None:
                _mac_pressure.set(mac["pressure_level"])
            self._bump_counters(mac)

        now = time.time()
        _last_sample.set(now)
        self.last_sample_ts = now
        self.latest = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlx": {
                "active_bytes": active,
                "cache_bytes": cache,
                "peak_bytes": peak,
                "active_gb": round(active / GB, 3),
            },
            "process": {"memory_bytes": proc_bytes, "source": source},
            "system": {
                "total_bytes": vm.total,
                "used_bytes": vm.used,
                "used_percent": vm.percent,
                "swap_used_bytes": swap.used,
            },
            "macos": mac,
        }

    def run(self):
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:  # keep the loop alive; surface via the counter
                _sample_errors.inc()
                print(f"sampler error: {exc!r}", file=sys.stderr)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def build_app(sampler, server_thread):
    app = FastAPI(title="mlx-memory-profiler", docs_url=None, redoc_url=None)

    @app.get("/metrics")
    def metrics():
        _server_up.set(1 if server_thread.is_alive() else 0)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz():
        alive = server_thread.is_alive()
        last = sampler.last_sample_ts
        age = time.time() - last if last else None
        healthy = alive and age is not None and age < max(30, sampler.interval * 5)
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "healthy": healthy,
                "model_server_alive": alive,
                "last_sample_age_seconds": round(age, 1) if age is not None else None,
            },
        )

    @app.get("/api/snapshot")
    def snapshot():
        if not sampler.latest:
            return JSONResponse(status_code=503, content={"detail": "no sample yet"})
        return sampler.latest

    return app


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-host", default="0.0.0.0", help="Bind host for the metrics endpoint (default: 0.0.0.0)")
    parser.add_argument("--metrics-port", type=int, default=9105, help="Port for the metrics endpoint (default: 9105)")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="Seconds between metric samples (default: 2.0)")
    parser.add_argument("--no-footprint", action="store_true", help="Use RSS instead of the `footprint` tool")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help and mlx_lm.server's help")
    args, server_args = parser.parse_known_args()

    if args.help:
        print(__doc__)
        print("\n--- mlx_lm.server flags (passed through) ---\n")
        sys.argv = ["mlx_lm.server", "--help"]
        try:
            mlx_server.main()
        except SystemExit:
            pass
        return

    gpu_info = device_memory_info()
    if gpu_info:
        _gpu_total.set(gpu_info["total_gb"] * GB)
        _gpu_working_set.set(gpu_info["recommended_working_set_gb"] * GB)
        print(
            f"GPU: {gpu_info['device_name']}  unified memory: {gpu_info['total_gb']:.1f} GB  "
            f"recommended working set: {gpu_info['recommended_working_set_gb']:.1f} GB"
        )

    use_footprint = HAVE_FOOTPRINT and not args.no_footprint
    if not use_footprint and not args.no_footprint:
        print("Note: `footprint` not found (xcode-select --install for accurate Apple Silicon numbers); using RSS.")

    _up.set(1)

    sys.argv = ["mlx_lm.server"] + server_args
    server_thread = threading.Thread(target=mlx_server.main, name="mlx_lm.server", daemon=True)
    server_thread.start()

    sampler = Sampler(args.sample_interval, use_footprint)
    sampler_thread = threading.Thread(target=sampler.run, name="sampler", daemon=True)
    sampler_thread.start()

    app = build_app(sampler, server_thread)
    print(f"metrics: http://{args.metrics_host}:{args.metrics_port}/metrics  (model server flags -> mlx_lm.server)")
    try:
        uvicorn.run(app, host=args.metrics_host, port=args.metrics_port, log_level="warning")
    finally:
        sampler.stop()
        _up.set(0)


if __name__ == "__main__":
    main()
