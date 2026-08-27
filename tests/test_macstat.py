"""macstat.py — vm_stat parsing and the Activity Monitor-style breakdown."""
import subprocess
from pathlib import Path

import macstat

FIXTURE = (Path(__file__).parent / "fixtures" / "vm_stat.txt").read_text()
PS = 16384  # page size in the fixture


def _fake_run(vm_stat_out=FIXTURE, pressure="1"):
    def run(cmd, *args, **kwargs):
        if cmd[0] == "vm_stat":
            return subprocess.CompletedProcess(cmd, 0, stdout=vm_stat_out, stderr="")
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(cmd, 0, stdout=pressure + "\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    return run


def test_parse_vm_stat_reads_page_size():
    page_size, _ = macstat._parse_vm_stat(FIXTURE)
    assert page_size == PS


def test_parse_vm_stat_reads_counts_including_quoted_label():
    _, raw = macstat._parse_vm_stat(FIXTURE)
    assert raw["wired_pages"] == 268246
    assert raw["compressor_pages"] == 44642
    assert raw["anonymous_pages"] == 1286884
    assert raw["purgeable_pages"] == 138454
    assert raw["file_backed_pages"] == 414135
    assert raw["pageins"] == 78874554
    assert raw["swapouts"] == 1527838
    # "Translation faults" carries surrounding quotes in vm_stat output
    assert raw["faults"] == 2867647168


def test_mac_memory_stats_breakdown(monkeypatch):
    monkeypatch.setattr(macstat, "HAVE_VM_STAT", True)
    monkeypatch.setattr(macstat.subprocess, "run", _fake_run())

    s = macstat.mac_memory_stats()

    assert s["page_size"] == PS
    assert s["wired_bytes"] == 268246 * PS
    assert s["compressed_bytes"] == 44642 * PS
    assert s["cached_files_bytes"] == 414135 * PS
    # App memory is anonymous minus purgeable
    assert s["app_bytes"] == (1286884 - 138454) * PS
    # "Memory Used" headline is wired + compressed + app
    assert s["used_bytes"] == s["wired_bytes"] + s["compressed_bytes"] + s["app_bytes"]
    # cumulative counters pass straight through
    assert s["faults"] == 2867647168
    assert s["pageins"] == 78874554


def test_mac_memory_stats_pressure_states(monkeypatch):
    monkeypatch.setattr(macstat, "HAVE_VM_STAT", True)

    monkeypatch.setattr(macstat.subprocess, "run", _fake_run(pressure="1"))
    assert macstat.mac_memory_stats()["pressure_state"] == "normal"

    monkeypatch.setattr(macstat.subprocess, "run", _fake_run(pressure="2"))
    assert macstat.mac_memory_stats()["pressure_state"] == "warning"

    monkeypatch.setattr(macstat.subprocess, "run", _fake_run(pressure="4"))
    s = macstat.mac_memory_stats()
    assert s["pressure_level"] == 4
    assert s["pressure_state"] == "critical"


def test_app_bytes_never_negative(monkeypatch):
    weird = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages wired down:  5.\n"
        "Pages purgeable:  99.\n"
        "Anonymous pages:  10.\n"
        "Pages occupied by compressor:  1.\n"
        "File-backed pages:  3.\n"
    )
    monkeypatch.setattr(macstat, "HAVE_VM_STAT", True)
    monkeypatch.setattr(macstat.subprocess, "run", _fake_run(vm_stat_out=weird))
    assert macstat.mac_memory_stats()["app_bytes"] == 0


def test_mac_memory_stats_none_without_vm_stat(monkeypatch):
    monkeypatch.setattr(macstat, "HAVE_VM_STAT", False)
    assert macstat.mac_memory_stats() is None


def test_mac_memory_stats_none_on_vm_stat_failure(monkeypatch):
    monkeypatch.setattr(macstat, "HAVE_VM_STAT", True)
    monkeypatch.setattr(
        macstat.subprocess, "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, "", ""),
    )
    assert macstat.mac_memory_stats() is None
