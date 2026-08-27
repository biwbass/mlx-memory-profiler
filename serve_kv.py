#!/usr/bin/env python3
"""
Run `mlx_lm.server` with KV-cache-aware memory tracking.

Unlike monitor.py (which watches a process from the outside), this loads
mlx_lm.server as a module and runs it in a background thread of *this*
process, so it can poll MLX's own allocator counters (mx.get_active_memory
/ get_cache_memory / get_peak_memory) directly. Those numbers include model
weights AND the KV cache together — there's no public API to isolate KV
cache alone — but since weights are static after load, the shape tells the
story: one big jump while the model loads, then smaller stepped increases
as requests grow the KV cache.

The chart also marks Apple's recommended GPU working-set ceiling (see
mlxinfo.py) — the same limit mlx_lm.server itself passes to
mx.set_wired_limit() on startup — as the practical "how much can MLX
actually use" number, alongside your --monitor-limit-gb (defaults to this
machine's total unified memory).

Usage: pass mlx_lm.server's own flags straight through, plus optional
--monitor-* flags for this script:

    python serve_kv.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

    python serve_kv.py --model ... --monitor-interval 0.5 --monitor-limit-gb 32

Requires: pip install psutil matplotlib mlx-lm

Caveat: relies on mlx_lm.server exposing a callable main() that reads
sys.argv, which is true as of the mlx-lm versions this was written against
but isn't a guaranteed public API — if a future mlx-lm release restructures
its CLI entry point, this may need updating.
"""
import argparse
import csv
import os
import sys
import tempfile
import threading
import time
from datetime import datetime

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

from chart import save_plot
from mlxinfo import device_memory_info
from procmem import HAVE_FOOTPRINT, tree_footprint_bytes, tree_procs, tree_rss_bytes


def _stat(*names):
    for name in names:
        fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
        if fn is not None:
            return fn()
    return None


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--monitor-limit-gb", type=float, default=None)
    parser.add_argument("--monitor-warn-pct", type=float, default=85.0)
    parser.add_argument("--monitor-out", default=None)
    parser.add_argument("--monitor-plot-out", default=None)
    parser.add_argument("--monitor-no-footprint", action="store_true")
    args, server_args = parser.parse_known_args()

    gpu_info = device_memory_info()  # mlx is a hard dependency here, so this should always resolve on Apple Silicon
    limit_gb = args.monitor_limit_gb or (gpu_info["total_gb"] if gpu_info else 32.0)

    use_footprint = HAVE_FOOTPRINT and not args.monitor_no_footprint
    proc_label = "footprint" if use_footprint else "RSS"
    fp_tmp = None
    if use_footprint:
        fp_fd, fp_tmp = tempfile.mkstemp(suffix=".json")
        os.close(fp_fd)

    out_path = args.monitor_out or f"mlx_kv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    sys.argv = ["mlx_lm.server"] + server_args
    server_thread = threading.Thread(target=mlx_server.main, daemon=True)
    server_thread.start()

    proc = psutil.Process()
    samples = []  # (elapsed_seconds, active_gb, cache_gb, peak_gb, proc_gb, sys_used_gb)
    t_start = time.monotonic()
    peak_active = 0
    warned_gpu = False

    print(f"Running mlx_lm.server in-process, sampling every {args.monitor_interval}s -> {out_path}")
    if gpu_info:
        print(f"GPU: {gpu_info['device_name']}  unified memory: {gpu_info['total_gb']:.1f} GB  "
              f"recommended working set: {gpu_info['recommended_working_set_gb']:.1f} GB")
    print("Ctrl+C to stop the server and save the chart.")

    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "active_gb", "cache_gb", "peak_gb", f"proc_{proc_label.lower()}_gb", "sys_used_gb", "sys_pct"])

            try:
                while server_thread.is_alive():
                    active = _stat("get_active_memory") or 0
                    cache = _stat("get_cache_memory") or 0
                    peak = _stat("get_peak_memory") or 0

                    procs = tree_procs(proc)
                    proc_bytes = tree_footprint_bytes(procs, fp_tmp) if use_footprint else None
                    if proc_bytes is None:
                        proc_bytes = tree_rss_bytes(procs)
                    vm = psutil.virtual_memory()

                    peak_active = max(peak_active, active)

                    active_gb, cache_gb, peak_gb = active / 1024 ** 3, cache / 1024 ** 3, peak / 1024 ** 3
                    proc_gb, sys_used_gb = proc_bytes / 1024 ** 3, vm.used / 1024 ** 3

                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        f"{active_gb:.3f}", f"{cache_gb:.3f}", f"{peak_gb:.3f}",
                        f"{proc_gb:.3f}", f"{sys_used_gb:.3f}", f"{vm.percent:.1f}",
                    ])
                    f.flush()
                    samples.append((time.monotonic() - t_start, active_gb, cache_gb, peak_gb, proc_gb, sys_used_gb))

                    print(
                        f"\rmlx: active={active_gb:6.2f} GB  cache={cache_gb:5.2f} GB  peak={peak_gb:6.2f} GB"
                        f"   |   proc({proc_label})={proc_gb:6.2f} GB  system={sys_used_gb:6.2f} GB ({vm.percent:4.1f}%)",
                        end="", flush=True,
                    )

                    if gpu_info and active_gb >= gpu_info["recommended_working_set_gb"] and not warned_gpu:
                        print(f"\n⚠ MLX active memory ({active_gb:.2f} GB) has passed the GPU's recommended "
                              f"working set ({gpu_info['recommended_working_set_gb']:.1f} GB)")
                        warned_gpu = True

                    time.sleep(args.monitor_interval)
            except KeyboardInterrupt:
                print("\nStopped by user")
    finally:
        if fp_tmp:
            try:
                os.unlink(fp_tmp)
            except OSError:
                pass

    print(f"\nPeak MLX active memory: {peak_active / 1024 ** 3:.2f} GB")
    if gpu_info:
        headroom = gpu_info["recommended_working_set_gb"] - peak_active / 1024 ** 3
        print(f"Headroom vs GPU recommended working set ({gpu_info['recommended_working_set_gb']:.1f} GB): "
              f"{headroom:.2f} GB")
    print(f"Log written to {out_path}")

    if samples:
        plot_path = args.monitor_plot_out or out_path.rsplit(".", 1)[0] + ".png"
        t = [s[0] for s in samples]
        hlines = {f"limit ({limit_gb:.0f} GB)": limit_gb}
        if gpu_info:
            hlines[f"GPU working set ({gpu_info['recommended_working_set_gb']:.1f} GB)"] = gpu_info["recommended_working_set_gb"]
        save_plot(
            t,
            series={
                "MLX active (weights+KV)": [s[1] for s in samples],
                "MLX cache": [s[2] for s in samples],
                f"process {proc_label}": [s[4] for s in samples],
                "system used": [s[5] for s in samples],
            },
            hlines=hlines,
            path=plot_path,
            title="mlx_lm.server memory (KV-aware)",
        )
        print(f"Chart written to {plot_path}")


if __name__ == "__main__":
    main()
