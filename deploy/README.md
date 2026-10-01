# Deploying the profiler service

[`profiler.py`](../profiler.py) supervises an OpenAI-compatible model server
(everything after `--`, e.g. `mlx-openai-server launch ...`) as a child process
and runs a FastAPI app exposing `/metrics` (Prometheus), `/healthz`, and
`/api/snapshot`.

```
Mac mini (macOS / Metal)                    k8s cluster (monitoring ns)
┌──────────────────────────────────┐        ┌──────────────────────────┐
│ profiler.py                      │        │ Prometheus               │
│  ├─ spawns: mlx-openai-server     │        │  ScrapeConfig → :9105    │
│  │           ├─ uvicorn    :8080  │◀───────┤ Grafana                  │
│  │           └─ model subproc     │  LAN   │  └─ "MLX memory" dash    │
│  ├─ /metrics               :9105  │        │     (ConfigMap sidecar)  │
│  └─ sampler: footprint(tree),     │        └──────────────────────────┘
│     vm_stat, scrape :8080         │
│       /v1/internal/memory         │
└──────────────────────────────────┘
```

The MLX allocator counters (`mlx_active/cache/peak_memory_bytes`) come from the
model server's `/v1/internal/memory` endpoint — `mlx-openai-server` loads the
model in a spawned subprocess, so those per-process counters can't be read from
`profiler.py` itself. Stock `mlx-openai-server` doesn't ship that endpoint yet;
without it those three gauges are absent and `mlx_profiler_mlx_memory_up` reads
0. Everything else works regardless. Spec + implementation for the endpoint:
[`mlx-openai-server-metrics.md`](mlx-openai-server-metrics.md).

## 1. Mac mini — run the service

```bash
uv sync --extra service
pip install mlx-openai-server        # the model server, installed separately
```

Then either run it directly:

```bash
uv run profiler.py -- mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-9B-6bit --host 0.0.0.0 --port 8080
```

or install the launchd agent so it starts at login and restarts on crash:

```bash
cp deploy/launchd/site.jordanthomas.mlx-profiler.plist ~/Library/LaunchAgents/
# edit the --model-path / ports in the plist first
launchctl load ~/Library/LaunchAgents/site.jordanthomas.mlx-profiler.plist
launchctl start site.jordanthomas.mlx-profiler
tail -f ~/Library/Logs/mlx-profiler.log
```

If the model server exits, `profiler.py` exits too (so launchd restarts the
pair and the k8s scrape target flips unhealthy).

Check it locally:

```bash
curl -s localhost:9105/metrics | grep mlx_gpu
curl -s localhost:9105/api/snapshot | python3 -m json.tool
```

`--metrics-host 0.0.0.0` is required for the cluster to reach it. Consider a
firewall rule / LAN-only exposure — neither `profiler.py` nor the model server
has auth on by default (but see below).

### Bearer-token auth on the metrics endpoint

Defence-in-depth alongside the LAN firewall rule / cluster NetworkPolicy. Set
`MLX_PROFILER_METRICS_TOKEN` (or pass `--metrics-token`) and `/metrics` and
`/api/snapshot` require `Authorization: Bearer <token>`; `/healthz` stays open
so k8s probes still work.

```bash
# generate once, keep it out of shell history / `ps` output
openssl rand -hex 32 > ~/.mlx-profiler-token
launchctl setenv MLX_PROFILER_METRICS_TOKEN "$(cat ~/.mlx-profiler-token)"

curl -s -H "Authorization: Bearer $(cat ~/.mlx-profiler-token)" localhost:9105/metrics | grep mlx_gpu
curl -s localhost:9105/metrics            # -> 401
```

For the launchd agent, add it to the plist's `EnvironmentVariables` dict rather
than `ProgramArguments` (keeps it off the process command line):

```xml
<key>MLX_PROFILER_METRICS_TOKEN</key>
<string>…64 hex chars…</string>
```

Cluster side (issue #71): store the same value as a Secret and reference it from
the `ScrapeConfig` / `ServiceMonitor` — `authorization.credentials` (bearer) or
the older `bearerTokenSecret`. See [`k8s/scrapeconfig.yaml`](k8s/scrapeconfig.yaml).

## 2. k8s — scrape + dashboard

Both manifests target the `monitoring` namespace and are labelled for the
kube-prometheus-stack operator / Grafana sidecar. Drop them into
`homelab-infra/apps/kube-prometheus-stack-config/manifests/` for ArgoCD:

| File | Purpose |
|---|---|
| [`k8s/scrapeconfig.yaml`](k8s/scrapeconfig.yaml) | `ScrapeConfig` pointing at the Mac mini `:9105`. **Edit the target** (`mac-mini.lan:9105`), confirm the `release:` label matches the Helm release, and uncomment the `authorization` block if a bearer token is set. |
| [`k8s/grafana-dashboard-configmap.yaml`](k8s/grafana-dashboard-configmap.yaml) | Dashboard, auto-loaded via the `grafana_dashboard: "1"` label. Generated from [`grafana/mlx-memory-dashboard.json`](grafana/mlx-memory-dashboard.json) by [`k8s/gen-dashboard-configmap.sh`](k8s/gen-dashboard-configmap.sh). |

Verify after sync:

```bash
kubectl -n monitoring get scrapeconfig mlx-profiler
# Prometheus UI → Status → Targets → mlx-profiler should be UP
# Grafana → Dashboards → Homelab → "MLX memory (Mac mini)"
```

## Metrics exported

| Metric | Meaning |
|---|---|
| `mlx_gpu_total_memory_bytes` | Total unified memory on the mini (the raw RAM spec) |
| `mlx_gpu_recommended_working_set_bytes` | **The GPU RAM limit** — Apple's recommended working-set ceiling, the exact value MLX servers pass to `mx.set_wired_limit()` |
| `mlx_active_memory_bytes` | MLX active allocation: model weights + KV cache (scraped from the model server's `/v1/internal/memory`; absent if unavailable) |
| `mlx_cache_memory_bytes` | MLX allocator cache (freed buffers kept for reuse) — same source |
| `mlx_peak_memory_bytes` | Peak MLX active memory since the model server started — same source |
| `mlx_process_memory_bytes{source}` | Model-server process-tree memory (`footprint`, else `rss`) |
| `macos_system_memory_{total,used}_bytes`, `..._used_percent` | psutil headline numbers |
| `macos_swap_{used,total}_bytes` | Swap |
| `macos_memory_{wired,compressed,app,anonymous,purgeable,cached_files,free,used}_bytes` | Activity Monitor "Memory" breakdown, from `vm_stat` |
| `macos_memory_pressure_level` | 1 normal / 2 warning / 4 critical (`kern.memorystatus_vm_pressure_level`) |
| `macos_vm_{pageins,pageouts,swapins,swapouts,compressions,decompressions}_pages_total`, `macos_vm_faults_total` | Cumulative paging counters — `rate()` them |
| `mlx_profiler_up`, `mlx_profiler_model_server_up`, `mlx_profiler_mlx_memory_up`, `mlx_profiler_last_sample_timestamp_seconds`, `mlx_profiler_sample_errors_total`, `mlx_profiler_mlx_memory_scrape_errors_total` | Liveness |
