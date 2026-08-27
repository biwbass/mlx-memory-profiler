"""Shared process-memory helpers for monitor.py and serve_kv.py.

On Apple Silicon, plain RSS undercounts MLX/Metal workloads because
GPU-shared unified-memory pages don't fully show up there. `footprint`
(Xcode command line tools) reports phys_footprint, the same number
Activity Monitor's "Memory" column shows, so it's preferred when available.
"""
import json
import shutil
import subprocess

import psutil

HAVE_FOOTPRINT = shutil.which("footprint") is not None


def tree_procs(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def tree_rss_bytes(procs):
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def tree_footprint_bytes(procs, tmp_path):
    pids = [str(p.pid) for p in procs]
    if not pids:
        return None
    try:
        result = subprocess.run(
            ["footprint", "--json", tmp_path, *pids],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        with open(tmp_path) as f:
            data = json.load(f)
        return sum(p.get("footprint", 0) for p in data.get("processes", []))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, FileNotFoundError):
        return None
