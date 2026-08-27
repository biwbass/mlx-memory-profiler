"""procmem.py — process-tree memory reading."""
import json
import subprocess

import psutil

import procmem


class FakeProc:
    def __init__(self, rss, pid=1234, alive=True):
        self._rss = rss
        self._alive = alive
        self.pid = pid

    def memory_info(self):
        if not self._alive:
            raise psutil.NoSuchProcess(self.pid)
        return type("mem", (), {"rss": self._rss})()


def test_tree_rss_sums_live_processes():
    assert procmem.tree_rss_bytes([FakeProc(100), FakeProc(200), FakeProc(50)]) == 350


def test_tree_rss_skips_vanished_processes():
    assert procmem.tree_rss_bytes([FakeProc(100), FakeProc(200, alive=False)]) == 100


def test_tree_footprint_parses_summed_json(monkeypatch, tmp_path):
    out = tmp_path / "fp.json"

    def fake_run(cmd, *args, **kwargs):
        assert cmd[:3] == ["footprint", "--json", str(out)]
        out.write_text(json.dumps({"processes": [{"footprint": 10}, {"footprint": 5}]}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(procmem.subprocess, "run", fake_run)
    assert procmem.tree_footprint_bytes([FakeProc(1, pid=1), FakeProc(1, pid=2)], str(out)) == 15


def test_tree_footprint_none_on_empty_pids():
    assert procmem.tree_footprint_bytes([], "unused.json") is None


def test_tree_footprint_none_on_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        procmem.subprocess, "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, "", "boom"),
    )
    assert procmem.tree_footprint_bytes([FakeProc(1)], str(tmp_path / "x.json")) is None


def test_tree_footprint_none_on_timeout(monkeypatch, tmp_path):
    def boom(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(procmem.subprocess, "run", boom)
    assert procmem.tree_footprint_bytes([FakeProc(1)], str(tmp_path / "x.json")) is None
