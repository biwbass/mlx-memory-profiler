"""Optional MLX/Metal device info — the actual GPU (unified memory) limits
for this machine, shared by monitor.py and serve_kv.py.

mx.device_info()["max_recommended_working_set_size"] is what MLX itself
uses: mlx_lm.server calls mx.set_wired_limit() with this exact value on
startup. It's Apple's recommended ceiling for GPU-resident (wired) memory —
below total unified memory, because the OS and other apps need headroom
too — so it's a more meaningful "GPU memory limit" than the machine's raw
RAM size.
"""


def device_memory_info():
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if not mx.metal.is_available():
        return None
    info = mx.device_info()
    total = info.get("memory_size", 0)
    working_set = info.get("max_recommended_working_set_size", 0)
    if not total or not working_set:
        return None
    return {
        "device_name": info.get("device_name", "unknown"),
        "total_gb": total / 1024 ** 3,
        "recommended_working_set_gb": working_set / 1024 ** 3,
    }
