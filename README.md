# MLX memory profiler

Built to answer one question: **running a local LLM via MLX on a 32GB Mac
mini, how close to the ceiling am I actually getting?**

## Files

| File | What it does |
|---|---|
| [`monitor.py`](monitor.py) | Watches a process from the *outside* (by PID, name match, or by launching a command). Works no matter how the model is served. Logs CSV, optional `--plot` PNG. |
| [`serve.sh`](serve.sh) | Thin wrapper: launches `mlx_lm.server` under `monitor.py` so you don't have to remember the `-- python -m mlx_lm.server ...` syntax. |
| [`serve_kv.py`](serve_kv.py) | Runs `mlx_lm.server` *in-process* (background thread) so it can poll MLX's own allocator counters directly — active/cache/peak memory, which includes weights + KV cache. Verified against a real model load, see Known Issues. |
| [`profiler.py`](profiler.py) | Same in-process hosting as `serve_kv.py`, but exposes a live `/metrics` endpoint (Prometheus) plus `/healthz` and `/api/snapshot` instead of a CSV+PNG at the end. Drop-in wrapper — pass `mlx_lm.server` flags straight through. For an always-on homelab setup scraped by Prometheus/Grafana; see [`deploy/`](deploy/). Needs `uv sync --extra service`. |
| [`macstat.py`](macstat.py) | Shared: full macOS memory picture from `vm_stat` + `sysctl` — Activity Monitor breakdown (wired/compressed/app/cached), memory-pressure level, paging counters. |
| [`mlx_mem.py`](mlx_mem.py) | Import-and-use helper for a custom inference script you write yourself (not `mlx_lm.server`) — same MLX counters, wrapped as a context manager. |
| `procmem.py` | Shared: process-tree memory reading (RSS vs. `footprint`). |
| `mlxinfo.py` | Shared: reads `mx.device_info()` for the GPU/unified-memory ceiling. |
| `chart.py` | Shared: generic multi-series PNG chart with reference lines. |

## Usage

```bash
# Attach to a server you already started
python monitor.py --name mlx_lm.server --plot

# Or launch it yourself
python monitor.py --plot -- python -m mlx_lm.server --model mlx-community/Qwen3.5-9B-6bit --port 8080

# Or the convenience wrapper (same thing, forwards flags to mlx_lm.server)
./serve.sh --model mlx-community/Qwen3.5-9B-6bit --port 8080

# Always-on: serve the model AND expose Prometheus metrics on :9105
uv sync --extra service
uv run profiler.py --model mlx-community/Qwen3.5-9B-6bit --port 8080
```

Needs `pip install psutil` (`matplotlib` too if using `--plot`).

## Development

```bash
uv sync --extra service --extra dev
uv run pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same on the homelab ARC runners.
MLX is Apple-silicon only, so `profiler.py` imports with `mx = None` off a
Mac and the model-server path is guarded in `main()` — the sampler maths,
the metric wiring and the FastAPI endpoints are all still covered.

## Design choices and why

**External monitor, not in-process, as the primary tool.** The model runs
as a separate process (`mlx_lm.server`, LM Studio, whatever) — not code we
write — so the only universally-working approach is watching it from
outside via `psutil`. `serve_kv.py` and `mlx_mem.py` exist for the cases
where you *can* get inside the process, but `monitor.py` is the one that
always works.

**`footprint` over plain RSS.** First version used `psutil`'s RSS. Tested
against a real Qwen3.5-9B run and it was off by roughly 2x from what
Activity Monitor showed (RSS ~5.8GB vs. Activity Monitor's ~12GB for the
same process). This is a known Apple Silicon issue: RSS (`task_basic_info`)
doesn't fully account for GPU-shared unified-memory pages, which is exactly
what MLX arrays live in. Fixed by shelling out to `footprint` (ships with
Xcode command line tools, install via `xcode-select --install`), which
reports `phys_footprint` — the same number Activity Monitor's "Memory"
column uses. `monitor.py` falls back to RSS with a printed note if
`footprint` isn't on PATH.

**GPU working-set ceiling, not just total RAM.** `mx.device_info()` exposes
`max_recommended_working_set_size` — Apple's recommended ceiling for
GPU-resident (wired) memory, below the machine's total unified memory
because the OS and other apps need headroom too. This is the *exact* value
`mlx_lm.server` itself passes to `mx.set_wired_limit()` on startup, so it's
a more meaningful "real limit" than the raw 32GB spec. Both `monitor.py`
and `serve_kv.py` auto-detect it (if MLX is importable in that environment)
and plot it as a reference line, and default `--limit-gb` to the detected
total unified memory instead of a hardcoded 32 if not passed explicitly.

**KV cache can't be isolated from weights.** MLX's `get_active_memory()`
counts everything currently allocated — model weights and KV cache
together — there's no public API to split them apart. Since weights are
static after load, the *shape* of the active-memory curve tells the story
instead: one big jump while the model loads, then smaller stepped
increases as requests grow the KV cache. `serve_kv.py`'s chart is built to
be read that way, not as a literal "KV cache = X GB" number.

**CSV + PNG, not a live dashboard.** Kept it to two flat output artifacts
per run rather than a web UI or curses TUI — simplest thing that lets you
eyeball a chart after a run or diff two runs' CSVs, no extra dependencies
beyond matplotlib.

## Known issues

- **`serve_kv.py` startup hang — resolved, was an mlx-lm version issue.**
  Originally reported hanging when starting `mlx_lm.server` in a background
  thread with no `--model` passed. Re-tested against `mlx-lm` 0.31.3 both
  without `--model` and with a real model load
  (`mlx-community/Qwen3.5-9B-MLX-8bit`): starts immediately, serves
  `/health` and `/v1/chat/completions` correctly, and `active`/`cache`
  counters track real usage (weights load as one jump, KV cache grows in
  small steps during a request). Not reproducible on this `mlx-lm`
  version — whatever blocked `ModelProvider` init originally appears to
  have been fixed upstream. If you hit a hang on an older `mlx-lm`,
  upgrading (`uv add mlx-lm` / `uv lock --upgrade-package mlx-lm`) is the
  first thing to try.

- **Background/detached processes may not respond to `Ctrl+C`/`SIGINT`.**
  If you launch `monitor.py` or `serve_kv.py` via `nohup ... &` and detach
  it from its shell (e.g. `disown`), `kill -INT <pid>` can leave it parked
  in its sample loop instead of triggering the `KeyboardInterrupt` handler
  — observed with both scripts. Normal foreground use (`Ctrl+C` in the
  terminal you launched it from) is unaffected; this only bites
  scripted/backgrounded runs. If it happens, `kill -TERM <pid>` stops it
  immediately — the CSV is safe either way since each row is flushed as
  it's written — but you'll lose the final peak-summary print and
  `--plot` chart (rebuild the chart from the CSV with `chart.save_plot`
  if needed).
