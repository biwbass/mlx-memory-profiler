# `/v1/internal/memory` — the endpoint `profiler.py` scrapes

`profiler.py` reads the MLX allocator counters (`mlx_active_memory_bytes`,
`mlx_cache_memory_bytes`, `mlx_peak_memory_bytes`) from an HTTP endpoint on the
model server rather than in-process, because
[`mlx-openai-server`](https://github.com/cubist38/mlx-openai-server) loads the
model in a **spawned subprocess** (to dodge an MLX Metal command-buffer race,
[ml-explore/mlx#2457](https://github.com/ml-explore/mlx/issues/2457)). Those
counters are per-process and only meaningful inside the process that holds the
model, so `profiler.py` can't read them directly and neither can the server's
own uvicorn parent.

Stock `mlx-openai-server` does not expose this yet. Until it does, the three
`mlx_*_memory_bytes` gauges are simply absent and `mlx_profiler_mlx_memory_up`
reads `0` — the footprint-vs-ceiling picture still works. This doc is the spec
for adding the endpoint (upstream PR, or a carried patch for the homelab
install in `bootstrap-device#34`).

## Contract

```
GET /v1/internal/memory   ->   200 {"active_bytes": <int>, "cache_bytes": <int>, "peak_bytes": <int>}
                               503 {"detail": "..."}   when the model isn't loaded
```

`profiler.py` also accepts the same three keys nested under a `"memory"` object,
and tolerates any subset. Override the path with `--mlx-memory-path`; disable
the scrape entirely with `--no-mlx-memory`.

## Implementation (3 changes)

### 1. Report the counters from the handler subprocess

`app/core/handler_process.py`, inside `_handler_worker._main._handle_request` —
`mlx.core` is already imported as `mx` at the top of `_handler_worker`, and this
runs in the child that owns the model. Special-case the method before the
generic `getattr(handler, method_name)` dispatch:

```python
async def _handle_request(request: dict[str, Any]) -> None:
    req_id: str = request.get("id", "")
    method_name: str = request.get("method", "")
    ...
    try:
        if method_name == "get_memory_snapshot":
            response_queue.put({"id": req_id, "type": "result", "value": {
                "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }})
            return
        method = getattr(handler, method_name)
        ...
```

### 2. Forward it through the proxy

`app/core/handler_process.py`, `HandlerProcessProxy`, next to `get_queue_stats`:

```python
async def get_memory_snapshot(self) -> dict[str, Any]:
    """MLX allocator counters from the handler subprocess."""
    return await self._call("get_memory_snapshot")
```

### 3. Expose the endpoint

`app/api/endpoints.py`, next to the `/v1/queue/stats` route:

```python
@router.get("/v1/internal/memory", response_model=None)
async def internal_memory(raw_request: Request) -> dict[str, Any] | JSONResponse:
    """MLX allocator counters (weights + KV cache together) for scraping."""
    handler = await _resolve_handler(raw_request)
    if handler is None:
        registry = getattr(raw_request.app.state, "registry", None)
        if registry is not None:
            for mid in registry.list_model_ids():
                try:
                    handler = registry.get_handler(mid)
                    break
                except KeyError:
                    continue
    if handler is not None and hasattr(handler, "get_memory_snapshot"):
        return await handler.get_memory_snapshot()

    # Single-process fallback: the model is in *this* process.
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"detail": "model not loaded"},
        )
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
```

`endpoints.py` doesn't import `sys` — add `import sys` to the module imports.

## Testing on Apple Silicon

```bash
mlx-openai-server launch --model-path mlx-community/Qwen3.5-9B-6bit --port 8080 &
curl -s localhost:8080/v1/internal/memory        # {"active_bytes": 6.1e9, ...} once loaded

uv run profiler.py -- mlx-openai-server launch --model-path ... --port 8080
curl -s localhost:9105/metrics | grep -E 'mlx_(active|cache|peak)_memory_bytes|mlx_profiler_mlx_memory_up'
```
