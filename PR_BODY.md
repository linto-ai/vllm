## Purpose

Make `Voxtral-Mini-4B-Realtime-2602` (and other models declaring uniform sliding window without explicit `layer_types`) actually usable on a 16 GB GPU for streaming workloads.

Today the model boots, accepts a session, and crashes after `max_model_len * 0.08` seconds with:

```
AssertionError: Sampled token IDs exceed the max model length.
Total number of tokens: N+1 > max_model_len: N
```

…taking the entire engine and every concurrent session down with it. The issue has been open since March (#38233) and has spawned several stalled fix attempts (#36089 closed unmerged, #40072 awaiting review, #39229 awaiting review). The PR below combines two small, targeted changes that together restore long-running stability **and** unlock real concurrency on small GPUs.

## What's broken

1. **`SlidingWindowSpec` never reaches the KV cache for Voxtral.**
   `VoxtralRealtimeGeneration` instantiates its language model from `text_config`, which declares `sliding_window: 8192` and `max_position_embeddings: 131072`. But `llama.py` only propagates `sliding_window` through the `layer_types` branch (Gemma-4-style hybrid). When `layer_types` is absent, the value is silently ignored and every layer is created with full attention.
   
   Consequences:
   - The KV cache pool is sized for `max_model_len`, not for `sliding_window` → on a 16 GB GPU you cannot fit even a single concurrent session at `max_model_len = 131072`.
   - The recycling-aware admission added by #40946 (`max_admission_blocks_per_request`) becomes a no-op for these models because the spec falls back to `FullAttentionSpec`.
   - Operators are forced to set `--max-model-len` aggressively low (typical: 4 000–16 000), which then…

2. **Streaming sessions crash hard the moment they reach `max_model_len`.**
   The `VoxtralRealtimeBuffer` keeps a constant-memory audio buffer, but the language model's KV cache and the scheduler's `num_computed_tokens` counter both grow monotonically. Once the running total touches `max_model_len`, `gpu_model_runner._bookkeeping_sync` fires a fatal assertion, the EngineCore dies, and every concurrent session is killed.

## Changes

### Commit 1 — `vllm/model_executor/models/llama.py`

Add an `elif` branch so a config that declares `sliding_window` without `layer_types` propagates uniformly to every attention layer (Mistral 7B v0.1/v0.2 lineage, Voxtral realtime).

```python
elif (cfg_sw := getattr(config, "sliding_window", None)) is not None:
    sliding_window = cfg_sw
```

7 new lines, no behavior change for any model that has `layer_types` (Gemma-4 etc.) or that does not declare `sliding_window` at all.

### Commit 2 — `vllm/v1/core/sched/scheduler.py`

Three guards that finish length-capped streaming sessions gracefully (`FINISHED_LENGTH_CAPPED`) instead of letting them crash the model runner:

- WAITING scheduling path: clamp `num_new_tokens` (mirroring the existing RUNNING-path guard) and finish requests whose clamp yields ≤ 0 instead of `assert num_new_tokens > 0`.
- `_handle_stopped_request`: after `_update_request_as_session`, finish if `request.num_tokens >= max_model_len`.
- `add_request`: same check on the `WAITING_FOR_STREAMING_REQ` path receiving a new chunk.

This part of the change mirrors the design in #40072 by @ianliuy. I bundled it here because in isolation it does not solve the 16 GB OOM problem (a healthy admission/sizing path is also required), and in isolation the sliding-window propagation does not solve the eventual `max_model_len` crash. They need to land together to make Voxtral realtime stable.

## Empirical validation

Tested on a single RTX A4000 16 GB (kube-linto-ai cluster, production deployment).

|                                  | Before | After |
|---|---|---|
| Streams concurrent stables       | 1–2    | **3–4** measured, 5+ theoretical |
| Pool KV per stream               | ~13 GiB at `max_model_len=131072` | **832 MiB** (`8192 × 104 KiB/token`) |
| Session ≥ 1 hour                 | Crash engine → all sessions die | Single session: stable, server-side LENGTH_CAPPED at `max_model_len`, pod stays up |
| Multi-stream behaviour at limit  | Engine dead → pod restart        | Length-capped sessions finish, others continue |

Reproduction:

```bash
docker run --gpus all -p 8000:8000 \
  ghcr.io/linto-ai/vllm:voxtral-sw \
  mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 64 \
  --gpu-memory-utilization 0.97 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

Where the same config without the patches OOMs at boot.

## What this does NOT solve

- **Per-session duration is still capped at `max_model_len * 0.08` seconds** (~2h54 with `max_model_len=131072`, the model's hard RoPE limit). To stream beyond this, callers must reconnect the WebSocket. We are exploring a server-side transparent re-anchor in a follow-up; happy to discuss design.
- The unrelated Voxtral realtime bugs #36015 (silent hang via unhandled `asyncio.TimeoutError`) and #34532 (disconnect crash) are out of scope for this PR — they have their own dedicated proposals.

## Tests

Existing tests pass locally (`pytest tests/v1/core/sched/`). I would like a reviewer's guidance on what to add specifically — happy to extend `tests/v1/streaming_input/test_scheduler_streaming.py` with the regression cases from #40072 (length-capped via `_handle_stopped_request`, via `add_request`, and the head-of-line-blocking test) and a `tests/models/multimodal/generation/test_voxtral_realtime_sliding_window.py` that asserts the spec is correctly created with `sliding_window=8192`. Let me know which scope you'd prefer landed in this PR vs follow-up.

## Related

- Closes (or partially closes) #38233
- Related to / supersedes design from #40072 (cc @ianliuy)
- Builds on #40946
- Out of scope but tracked: #36015, #34532, #39229

## Backward compatibility

- Models with `layer_types` (Gemma-4, etc.): unchanged.
- Models without `sliding_window` in config: unchanged.
- Models with uniform `sliding_window` (Mistral 7B v0.1/v0.2, Voxtral realtime, Voxtral classic): now respect their own config — this is the intended behavior. If a model historically declared `sliding_window` in its config but was *not* actually trained to use it, reviewers please flag — I am unaware of such a case but the change is conservative enough to revert easily.
