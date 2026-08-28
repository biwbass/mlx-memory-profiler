#!/usr/bin/env python3
"""
Supervise an OpenAI-compatible MLX model server and expose Prometheus metrics.

`profiler.py` launches the model server you give it (everything after `--`)
as a child process, then runs a small FastAPI app that exposes `/metrics` in
Prometheus exposition format plus `/healthz` and `/api/snapshot`. A Prometheus
server (e.g. on a k8s cluster that can reach this Mac over the LAN) scrapes
`/metrics`; a Grafana dashboard graphs it. See `deploy/`.

    python profiler.py -- mlx-openai-server launch \
        --model-path mlx-community/Qwen3.5-9B-6bit --port 8080

    # model server -> :8080 (OpenAI-compatible, unchanged)
    # metrics      -> :9105/metrics

    # tune the sampler / move the metrics port / require a bearer token:
    MLX_PROFILER_METRICS_TOKEN=$(openssl rand -hex 32) \
        python profiler.py --metrics-port 9105 --sample-interval 2 -- \
        mlx-openai-server launch --model-path ... --port 8080

What the sampler reads every interval:

  * The child's whole process tree — phys_footprint via `footprint` (the
    number Activity Monitor shows), else RSS (procmem.py). `mlx-openai-server`
    loads the model in a *spawned subprocess*, so the tree walk is what
    captures the model's real memory.
  * The full macOS memory picture — Activity Monitor breakdown, memory
    pressure, paging counters (macstat.py).
  * MLX allocator counters (active / cache / peak) — scraped over HTTP from
    the model server's `/v1/internal/memory` endpoint, because those counters
    are per-process and only meaningful inside the process that holds the
    model. Needs a server build that exposes that endpoint; when it's absent
    or unreachable the `mlx_*_memory_bytes` gauges are simply not emitted
    (like off Apple Silicon) and `mlx_profiler_mlx_memory_up` reads 0.

Requires: pip install psutil fastapi "uvicorn[standard]" prometheus-client
(`uv sync --extra service`). The model server itself is whatever you pass
after `--`; it is not imported here.
"""
import argparse
import atexit
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    sys.exit("This script requires psutil: pip install psutil")

try:
    import uvicorn
    from fastapi import Depends, FastAPI, Header, HTTPException
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

# Bearer token for /metrics and /api/snapshot. CLI flag wins; this is the
# fallback so the launchd plist / k8s can inject it without it showing in `ps`.
TOKEN_ENV = "MLX_PROFILER_METRICS_TOKEN"

# Path on the model server that reports its MLX allocator counters as JSON
# ({"active_bytes", "cache_bytes", "peak_bytes"} — a nested "memory" object
# with the same keys is also accepted).
DEFAULT_MLX_MEMORY_PATH = "/v1/internal/memory"

# --- metrics ---------------------------------------------------------------

# MLX allocator counters (weights + KV cache together — MLX has no public API
# to split them). Scraped from the model server; see module docstring.
_mlx_active = Gauge("mlx_active_memory_bytes", "MLX active memory: model weights + KV cache")
_mlx_cache = Gauge("mlx_cache_memory_bytes", "MLX allocator cache (freed buffers held for reuse)")
_mlx_peak = Gauge("mlx_peak_memory_bytes", "MLX peak active memory since the model server started")

# The supervised server's process tree, via `footprint` (phys_footprint) when
# available, else RSS. Includes mlx-openai-server's spawned model subprocess.
_proc_mem = Gauge("mlx_process_memory_bytes", "Model-server process-tree memory", ["source"])

# Static ceilings from mx.device_info() (see mlxinfo.py).
_gpu_total = Gauge("mlx_gpu_total_memory_bytes", "Total unified memory on this machine")
_gpu_working_set = Gauge(
    "mlx_gpu_recommended_working_set_bytes",
    "Apple's recommended GPU working-set ceiling (the wired limit MLX servers set)",
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
_server_up = Gauge("mlx_profiler_model_server_up", "1 while the supervised model-server process is alive")
_mlx_mem_up = Gauge("mlx_profiler_mlx_memory_up", "1 when the model server's MLX memory endpoint was last scraped OK")
_last_sample = Gauge("mlx_profiler_last_sample_timestamp_seconds", "Unix time of the last successful sample")
_sample_errors = Counter("mlx_profiler_sample_errors_total", "Sampler iterations that raised")
_mlx_mem_errors = Counter(
    "mlx_profiler_mlx_memory_scrape_errors_total",
    "Failed scrapes of the model server's MLX memory endpoint",
)


def _counter_delta(prev, value):
    """How much to add to a Prometheus counter, given the previous and current
    reading of a cumulative kernel counter. The first reading (prev is None)
    contributes nothing — we adopt it as the baseline rather than emit the
    machine's lifetime total. A drop means the kernel counter reset (reboot),
    so the new value is itself the delta."""
    if prev is None:
        return 0
    if value >= prev:
        return value - prev
    return value


def scrape_mlx_memory(url, timeout=1.5):
    """GET `url` and pull MLX allocator counters out of the JSON body. Returns
    a dict with int `active`/`cache`/`peak` (any missing key omitted), or None
    if the request failed or the body wasn't usable. Never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (operator-supplied localhost URL)
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None

    body = payload.get("memory", payload) if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        return None

    out = {}
    for short, keys in (
        ("active", ("active_bytes", "active", "active_memory")),
        ("cache", ("cache_bytes", "cache", "cache_memory")),
        ("peak", ("peak_bytes", "peak", "peak_memory")),
    ):
        for key in keys:
            if isinstance(body.get(key), (int, float)):
                out[short] = int(body[key])
                break
    return out or None


def _server_alive(server):
    """True if the supervised server handle looks alive. Accepts a Popen
    (`.poll()`), a thread-like (`.is_alive()`), or None (treated as alive so
    the endpoints are testable without a real child)."""
    if server is None:
        return True
    if hasattr(server, "poll"):
        return server.poll() is None
    if hasattr(server, "is_alive"):
        return server.is_alive()
    return True


class Sampler:
    """Background loop: read every source once per interval, update metrics,
    and keep the latest reading around for the /api/snapshot endpoint."""

    def __init__(self, interval, use_footprint, target_pid=None, mlx_memory_url=None):
        self.interval = interval
        self.use_footprint = use_footprint
        self.mlx_memory_url = mlx_memory_url
        # The process to walk. Defaults to this process (handy for tests);
        # main() points it at the supervised model server.
        self.proc = psutil.Process(target_pid) if target_pid else psutil.Process()
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
            delta = _counter_delta(self._page_prev.get(key), value)
            if delta:
                counter.inc(delta)
            self._page_prev[key] = value

    def _sample_mlx_memory(self):
        """Scrape the model server's MLX counters. Returns the dict from
        scrape_mlx_memory (or None) and updates the related metrics."""
        if not self.mlx_memory_url:
            return None
        mem = scrape_mlx_memory(self.mlx_memory_url)
        if mem is None:
            _mlx_mem_errors.inc()
            _mlx_mem_up.set(0)
            return None
        if "active" in mem:
            _mlx_active.set(mem["active"])
        if "cache" in mem:
            _mlx_cache.set(mem["cache"])
        if "peak" in mem:
            _mlx_peak.set(mem["peak"])
        _mlx_mem_up.set(1)
        return mem

    def sample_once(self):
        mlx_mem = self._sample_mlx_memory()

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
        if mlx_mem is not None:
            active = mlx_mem.get("active")
            mlx_block = {
                "available": True,
                "active_bytes": active,
                "cache_bytes": mlx_mem.get("cache"),
                "peak_bytes": mlx_mem.get("peak"),
                "active_gb": round(active / GB, 3) if active is not None else None,
            }
        else:
            mlx_block = {"available": False}
        self.latest = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlx": mlx_block,
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


def _auth_dependency(token):
    """A FastAPI dependency that enforces `Authorization: Bearer <token>`.
    Returns None when no token is configured, so the endpoints stay open."""
    if not token:
        return None

    def require_bearer_token(authorization: str = Header(default="")):
        scheme, _, presented = authorization.partition(" ")
        # compare_digest keeps the check constant-time; encode so a non-ASCII
        # header can't raise instead of just failing.
        ok = scheme.lower() == "bearer" and secrets.compare_digest(
            presented.encode(), token.encode()
        )
        if not ok:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_bearer_token


def build_app(sampler, server, token=None):
    app = FastAPI(title="mlx-memory-profiler", docs_url=None, redoc_url=None)

    guard = _auth_dependency(token)
    protected = [Depends(guard)] if guard is not None else []

    @app.get("/metrics", dependencies=protected)
    def metrics():
        _server_up.set(1 if _server_alive(server) else 0)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz():
        alive = _server_alive(server)
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

    @app.get("/api/snapshot", dependencies=protected)
    def snapshot():
        if not sampler.latest:
            return JSONResponse(status_code=503, content={"detail": "no sample yet"})
        return sampler.latest

    return app


def _derive_server_url(explicit, server_cmd):
    """Where to reach the model server for scraping. Uses --server-url if
    given, else reads --host/--port out of the server command (0.0.0.0 ->
    127.0.0.1), else defaults to http://127.0.0.1:8000."""
    if explicit:
        return explicit.rstrip("/")
    host, port = "127.0.0.1", "8000"
    for i, tok in enumerate(server_cmd):
        for flag, setter in (("--host", "host"), ("--port", "port")):
            if tok == flag and i + 1 < len(server_cmd):
                val = server_cmd[i + 1]
            elif tok.startswith(flag + "="):
                val = tok.split("=", 1)[1]
            else:
                continue
            if setter == "host":
                host = val
            else:
                port = val
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _terminate(proc, timeout=10):
    """SIGTERM the child, then SIGKILL if it doesn't go."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-host", default="0.0.0.0", help="Bind host for the metrics endpoint (default: 0.0.0.0)")
    parser.add_argument("--metrics-port", type=int, default=9105, help="Port for the metrics endpoint (default: 9105)")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="Seconds between metric samples (default: 2.0)")
    parser.add_argument("--no-footprint", action="store_true", help="Use RSS instead of the `footprint` tool")
    parser.add_argument(
        "--server-url",
        default=None,
        help="Base URL to reach the model server for scraping (default: derived from its --host/--port, else http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--mlx-memory-path",
        default=DEFAULT_MLX_MEMORY_PATH,
        help=f"Path on the model server that reports MLX allocator counters (default: {DEFAULT_MLX_MEMORY_PATH})",
    )
    parser.add_argument(
        "--no-mlx-memory",
        action="store_true",
        help="Don't scrape the model server for MLX counters (the mlx_*_memory_bytes gauges won't be emitted)",
    )
    parser.add_argument(
        "--metrics-token",
        default=os.environ.get(TOKEN_ENV),
        help=(
            "Require 'Authorization: Bearer <token>' on /metrics and /api/snapshot "
            f"(default: ${TOKEN_ENV}; /healthz stays open for k8s probes)"
        ),
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show this help")
    args, server_cmd = parser.parse_known_args()

    if server_cmd and server_cmd[0] == "--":
        server_cmd = server_cmd[1:]

    if args.help:
        print(__doc__)
        return
    if not server_cmd:
        sys.exit(
            "error: no model-server command given. Put it after `--`, e.g.\n"
            "  python profiler.py -- mlx-openai-server launch --model-path <repo> --port 8080\n"
            "(--help for details)"
        )

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

    mlx_memory_url = None
    if not args.no_mlx_memory:
        mlx_memory_url = _derive_server_url(args.server_url, server_cmd) + args.mlx_memory_path

    _up.set(1)

    print(f"launching model server: {' '.join(server_cmd)}")
    proc = subprocess.Popen(server_cmd)  # noqa: S603 (operator-supplied command)

    sampler = Sampler(
        args.sample_interval, use_footprint,
        target_pid=proc.pid, mlx_memory_url=mlx_memory_url,
    )
    sampler_thread = threading.Thread(target=sampler.run, name="sampler", daemon=True)
    sampler_thread.start()

    # If the model server dies, take the profiler down with it so launchd /
    # the k8s scrape target flips unhealthy and the pair gets restarted.
    def _watch_child():
        proc.wait()
        print(f"model server exited (code {proc.returncode}); shutting down", file=sys.stderr)
        _server_up.set(0)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_watch_child, name="child-watch", daemon=True).start()

    app = build_app(sampler, proc, token=args.metrics_token)
    auth_note = "bearer-token auth on" if args.metrics_token else "no auth — keep it LAN-only"
    mem_note = mlx_memory_url or "disabled"
    print(
        f"metrics: http://{args.metrics_host}:{args.metrics_port}/metrics  "
        f"({auth_note}; MLX counters <- {mem_note})"
    )
    try:
        uvicorn.run(app, host=args.metrics_host, port=args.metrics_port, log_level="warning")
    finally:
        sampler.stop()
        _up.set(0)
        _terminate(proc)


if __name__ == "__main__":
    main()
