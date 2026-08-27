"""macOS system memory detail, parsed from `vm_stat` and `sysctl`.

`psutil.virtual_memory()` gives the headline used/free/percent numbers, but
not the Activity Monitor breakdown (wired / compressed / app / cached files),
the memory-pressure level, or the paging counters that show active thrashing.
Those only come from `vm_stat` page counts and a couple of sysctls, so this
module shells out for them — same approach `procmem.py` takes with `footprint`.

Everything here is best-effort: on a non-macOS host, or if `vm_stat` isn't on
PATH, `mac_memory_stats()` returns None and callers should carry on with the
psutil numbers alone.
"""
import re
import resource
import shutil
import subprocess

HAVE_VM_STAT = shutil.which("vm_stat") is not None

# vm_stat line label -> friendly key. Labels are matched after stripping the
# surrounding quotes some of them carry ("Translation faults").
_VM_STAT_KEYS = {
    "Pages free": "free_pages",
    "Pages active": "active_pages",
    "Pages inactive": "inactive_pages",
    "Pages speculative": "speculative_pages",
    "Pages wired down": "wired_pages",
    "Pages purgeable": "purgeable_pages",
    "Anonymous pages": "anonymous_pages",
    "File-backed pages": "file_backed_pages",
    "Pages occupied by compressor": "compressor_pages",
    "Pageins": "pageins",
    "Pageouts": "pageouts",
    "Swapins": "swapins",
    "Swapouts": "swapouts",
    "Compressions": "compressions",
    "Decompressions": "decompressions",
    "Translation faults": "faults",
}


def _pressure_level():
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        pass
    return None


PRESSURE_LABELS = {1: "normal", 2: "warning", 4: "critical"}


def _parse_vm_stat(text):
    page_size = resource.getpagesize()
    m = re.search(r"page size of (\d+) bytes", text)
    if m:
        page_size = int(m.group(1))

    raw = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.rsplit(":", 1)
        label = label.strip().strip('"')
        value = value.strip().rstrip(".")
        key = _VM_STAT_KEYS.get(label)
        if key and value.isdigit():
            raw[key] = int(value)
    return page_size, raw


def mac_memory_stats():
    """Activity Monitor-style memory breakdown plus paging counters.

    Returns a dict of byte values and cumulative page counters, or None if
    this isn't macOS / `vm_stat` isn't available.

    Byte fields (`*_bytes`) mirror Activity Monitor's "Memory" tab:
      wired / compressed  - exact
      app                 - anonymous minus purgeable (AM "App Memory")
      cached_files        - file-backed pages (AM "Cached Files", approx)
      used                - wired + compressed + app (AM "Memory Used")
    Counter fields are the raw cumulative `vm_stat` totals (pages, not bytes);
    export them as Prometheus counters and let the query engine rate() them.
    """
    if not HAVE_VM_STAT:
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None

    page_size, p = _parse_vm_stat(out.stdout)
    if "wired_pages" not in p:
        return None

    def b(key):
        return p.get(key, 0) * page_size

    anonymous = b("anonymous_pages")
    purgeable = b("purgeable_pages")
    wired = b("wired_pages")
    compressed = b("compressor_pages")
    app = max(anonymous - purgeable, 0)

    level = _pressure_level()
    return {
        "page_size": page_size,
        "wired_bytes": wired,
        "compressed_bytes": compressed,
        "anonymous_bytes": anonymous,
        "purgeable_bytes": purgeable,
        "app_bytes": app,
        "cached_files_bytes": b("file_backed_pages"),
        "free_bytes": b("free_pages"),
        "used_bytes": wired + compressed + app,
        "pressure_level": level,
        "pressure_state": PRESSURE_LABELS.get(level, "unknown"),
        # cumulative counters (pages)
        "pageins": p.get("pageins", 0),
        "pageouts": p.get("pageouts", 0),
        "swapins": p.get("swapins", 0),
        "swapouts": p.get("swapouts", 0),
        "compressions": p.get("compressions", 0),
        "decompressions": p.get("decompressions", 0),
        "faults": p.get("faults", 0),
    }


if __name__ == "__main__":
    import json

    stats = mac_memory_stats()
    if stats is None:
        print("vm_stat not available on this host")
    else:
        print(json.dumps(stats, indent=2))
