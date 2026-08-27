"""The modules that are meant to import without mlx / off macOS do."""


def test_mlx_optional_modules_import():
    import chart  # noqa: F401
    import macstat  # noqa: F401
    import mlxinfo  # noqa: F401
    import monitor  # noqa: F401
    import procmem  # noqa: F401
    import profiler  # noqa: F401


def test_helpers_degrade_without_platform_support():
    import macstat
    import mlxinfo

    # both return None rather than raising when their platform tool is absent
    mlxinfo.device_memory_info()
    macstat.mac_memory_stats()
