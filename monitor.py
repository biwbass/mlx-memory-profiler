#!/usr/bin/env python3
"""
Monitor memory usage (process + system) while an MLX workload runs.

Works no matter how you're serving the model — mlx_lm.server, a custom
inference script, LM Studio, etc. — because it watches the process tree and
system memory from the outside.

Usage:
    # Attach to a process you already started:
    python monitor.py --pid 12345

    # Or match by name/cmdline substring:
    python monitor.py --name mlx_lm.server

    # Or launch and monitor a command directly:
    python monitor.py -- python -m mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --prompt "hello" --max-tokens 200

Writes a CSV log and prints a live status line. On exit (Ctrl+C, or when the
launched command finishes) prints peak usage against --limit-gb (default:
auto-detected total unified memory if mlx is importable here, else 32 GB).

Pass --plot to also save a PNG chart of the run (proc footprint, system
used, and swap over time), with reference lines for --limit-gb, --warn-pct,
and — if this environment has mlx installed — Apple's recommended GPU
working-set ceiling (see mlxinfo.py): the same value mlx_lm.server itself
uses as its wired-memory limit, and a more meaningful "how much can MLX
actually use" number than raw RAM size.

Requires: pip install psutil   (and matplotlib if you use --plot)

Memory accounting: on Apple Silicon, plain RSS (what psutil reports) can
undercount by a wide margin for MLX/Metal workloads, because GPU-shared
unified-memory pages don't fully show up there. This script shells out to
`footprint` (ships with Xcode command line tools) each sample to get
phys_footprint — the same number Activity Monitor's "Memory" column shows —
and uses that as the headline figure, falling back to RSS with a warning if
`footprint` isn't on PATH. Install it with: xcode-select --install
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    sys.exit("This script requires psutil: pip install psutil")

from mlxinfo import device_memory_info
from procmem import HAVE_FOOTPRINT, tree_footprint_bytes, tree_procs, tree_rss_bytes


def find_pid_by_name(name_substr):
    needle = name_substr.lower()
    matches = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        name = p.info["name"] or ""
        cmdline = " ".join(p.info["cmdline"] or [])
        if needle in name.lower() or needle in cmdline.lower():
            matches.append(p)
    if not matches:
        sys.exit(f"No running process matched '{name_substr}'")
    if len(matches) > 1:
        pids = ", ".join(str(p.info["pid"]) for p in matches)
        sys.exit(f"Multiple processes matched '{name_substr}' (pids: {pids}); use --pid instead")
    return matches[0].info["pid"]


def save_plot(samples, plot_path, limit_gb, warn_pct, gpu_info, label):
    from chart import save_plot as _save_plot

    t = [s[0] for s in samples]
    series = {
        f"process {label} (tree)": [s[1] for s in samples],
        "system used": [s[2] for s in samples],
        "swap used": [s[3] for s in samples],
    }
    hlines = {
        f"limit ({limit_gb:.0f} GB)": limit_gb,
        f"warn ({warn_pct:.0f}%)": limit_gb * warn_pct / 100,
    }
    if gpu_info:
        hlines[f"GPU working set ({gpu_info['recommended_working_set_gb']:.1f} GB)"] = gpu_info["recommended_working_set_gb"]
    _save_plot(t, series=series, hlines=hlines, path=plot_path, title="MLX memory usage")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--pid", type=int, help="PID of an already-running process to monitor")
    target.add_argument("--name", help="Substring match against process name/cmdline")
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="Command to launch and monitor, given after --",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples (default: 1.0)")
    parser.add_argument(
        "--limit-gb", type=float, default=None,
        help="System memory to compare against (default: auto-detected unified memory, else 32)",
    )
    parser.add_argument("--warn-pct", type=float, default=85.0, help="Warn once system memory used exceeds this percent")
    parser.add_argument("--out", default=None, help="CSV output path (default: mlx_mem_<timestamp>.csv)")
    parser.add_argument("--plot", action="store_true", help="Save a PNG chart of the run alongside the CSV")
    parser.add_argument("--plot-out", default=None, help="PNG output path (default: same name as --out, .png)")
    parser.add_argument(
        "--no-footprint", action="store_true",
        help="Skip the `footprint` tool and use plain RSS (faster sampling, less accurate on Apple Silicon)",
    )
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command

    if command and (args.pid or args.name):
        sys.exit("Specify a command to launch, or --pid/--name to attach to an existing process, not both")

    child = None
    if command:
        child = subprocess.Popen(command)
        proc = psutil.Process(child.pid)
    elif args.pid:
        proc = psutil.Process(args.pid)
    elif args.name:
        proc = psutil.Process(find_pid_by_name(args.name))
    else:
        sys.exit("Specify --pid, --name, or a command to launch after --")

    gpu_info = device_memory_info()
    if args.limit_gb is None:
        args.limit_gb = gpu_info["total_gb"] if gpu_info else 32.0

    use_footprint = HAVE_FOOTPRINT and not args.no_footprint
    if not use_footprint and not args.no_footprint:
        print("Note: `footprint` tool not found (install Xcode command line tools for accurate "
              "Apple Silicon numbers: xcode-select --install). Falling back to RSS.")
    label = "footprint" if use_footprint else "RSS"

    out_path = args.out or f"mlx_mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fp_tmp = None
    if use_footprint:
        fp_fd, fp_tmp = tempfile.mkstemp(suffix=".json")
        os.close(fp_fd)

    peak_proc = 0
    peak_sys_used_gb = 0.0
    peak_sys_pct = 0.0
    warned = False
    warned_gpu = False
    samples = []  # (elapsed_seconds, proc_gb, sys_used_gb, swap_gb)
    t_start = time.monotonic()

    print(f"Monitoring PID {proc.pid} every {args.interval}s -> {out_path}  "
          f"(limit: {args.limit_gb:.0f} GB, proc metric: {label})")
    if gpu_info:
        print(f"GPU: {gpu_info['device_name']}  unified memory: {gpu_info['total_gb']:.1f} GB  "
              f"recommended working set: {gpu_info['recommended_working_set_gb']:.1f} GB")
    else:
        print("Note: mlx not importable here, so no GPU working-set ceiling to compare against "
              "(this only needs to be true in the environment running monitor.py, not the model).")

    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", f"proc_{label.lower()}_gb", "sys_used_gb", "sys_pct", "swap_used_gb"])

            try:
                while True:
                    if child is not None and child.poll() is not None:
                        print(f"\nMonitored command exited with code {child.returncode}")
                        break
                    if not proc.is_running():
                        print("\nMonitored process ended")
                        break

                    procs = tree_procs(proc)
                    proc_bytes = None
                    if use_footprint:
                        proc_bytes = tree_footprint_bytes(procs, fp_tmp)
                    if proc_bytes is None:
                        proc_bytes = tree_rss_bytes(procs)
                    vm = psutil.virtual_memory()
                    swap = psutil.swap_memory()

                    proc_gb = proc_bytes / (1024 ** 3)
                    sys_used_gb = vm.used / (1024 ** 3)
                    swap_gb = swap.used / (1024 ** 3)

                    peak_proc = max(peak_proc, proc_bytes)
                    peak_sys_used_gb = max(peak_sys_used_gb, sys_used_gb)
                    peak_sys_pct = max(peak_sys_pct, vm.percent)

                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        f"{proc_gb:.3f}", f"{sys_used_gb:.3f}", f"{vm.percent:.1f}", f"{swap_gb:.3f}",
                    ])
                    f.flush()
                    samples.append((time.monotonic() - t_start, proc_gb, sys_used_gb, swap_gb))

                    flag = "!" if vm.percent >= args.warn_pct else " "
                    print(
                        f"\r[{flag}] proc({label})={proc_gb:6.2f} GB  system={sys_used_gb:6.2f} GB "
                        f"({vm.percent:4.1f}%)  swap={swap_gb:5.2f} GB",
                        end="", flush=True,
                    )

                    if vm.percent >= args.warn_pct and not warned:
                        print(f"\n⚠ system memory at {vm.percent:.1f}% (>= {args.warn_pct:.0f}% warning threshold)")
                        warned = True

                    if gpu_info and proc_gb >= gpu_info["recommended_working_set_gb"] and not warned_gpu:
                        print(f"\n⚠ process {label} ({proc_gb:.2f} GB) has passed the GPU's recommended "
                              f"working set ({gpu_info['recommended_working_set_gb']:.1f} GB) — MLX may be "
                              f"leaning on swap/compression from here")
                        warned_gpu = True

                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped by user")
                if child is not None and child.poll() is None:
                    child.terminate()
    finally:
        if fp_tmp:
            try:
                os.unlink(fp_tmp)
            except OSError:
                pass

    print(f"\nPeak process {label}: {peak_proc / (1024 ** 3):.2f} GB")
    print(f"Peak system usage:  {peak_sys_used_gb:.2f} GB of {args.limit_gb:.0f} GB  (peak {peak_sys_pct:.1f}% of total system memory)")
    print(f"Headroom at peak:   {args.limit_gb - peak_sys_used_gb:.2f} GB")
    if gpu_info:
        headroom = gpu_info["recommended_working_set_gb"] - peak_proc / (1024 ** 3)
        print(f"Headroom vs GPU recommended working set ({gpu_info['recommended_working_set_gb']:.1f} GB): "
              f"{headroom:.2f} GB")
    print(f"Log written to {out_path}")

    if args.plot:
        if not samples:
            print("No samples collected; skipping plot")
        else:
            plot_path = args.plot_out or out_path.rsplit(".", 1)[0] + ".png"
            save_plot(samples, plot_path, args.limit_gb, args.warn_pct, gpu_info, label)
            print(f"Chart written to {plot_path}")


if __name__ == "__main__":
    main()
