# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blank-run penalty for realtime streaming: worker-side implementation.

Streaming speech-to-text models that re-ingest their own output can fall into
a self-sustained decoding rut: once the visible context fills with the
blank/silence token, greedy decoding keeps emitting it over real speech for
minutes (observed with Voxtral realtime; same family as Whisper repetition
loops). Temperature does not help: the distribution collapses to P(blank)~1.

Once a request has sampled the blank token more than K consecutive times, a
progressive penalty is subtracted from that token's logit before sampling:

    penalty = min(cap, alpha * (run_length - K))

Healthy blank runs (inter-sentence silences) stay below K and are never
touched; genuinely silent audio keeps decoding as silence because its blank
margin exceeds `cap`.

This lives in the model runner, NOT in a v1 LogitsProcessor: the realtime
streaming path recycles the engine request for every audio chunk and clears
its output-token list in place, so the `output_tok_ids` reference a logits
processor receives via BatchUpdate is empty at every step. The run length
must be accumulated worker-side, keyed by request id, which is stable for
the whole streaming session.

Per-request opt-in via SamplingParams.extra_args["blank_run_penalty"]
(a dict with token_id, k, alpha, cap), wired by the realtime connection
from the --realtime-blank-run-* engine flags; inert otherwise.

abort_after > 0: end the session after that many penalty-broken runs in
a row. A run > k only breaks when the penalty overcomes the blank margin
(stuck session) or when speech resumes after a long pause; the latter is
followed by real tokens, the former is not. True silence never breaks
(margin > cap), so this cannot fire on silent audio. Flagged req_ids
travel via ModelRunnerOutput.blank_run_aborts; the scheduler finishes
them like a max_model_len cap and the client reconnects.
"""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

EXTRA_ARGS_KEY = "blank_run_penalty"

# A stuck session emits ~1 junk token per broken run; resumed speech emits a
# stream of them. More real tokens than this between two breaks ends the streak.
ABORT_MAX_TOKENS_BETWEEN = 3


@dataclass(frozen=True)
class BlankRunConfig:
    token_id: int
    k: int
    alpha: float
    cap: float
    abort_after: int = 0  # 0 = never abort


def parse_config(sampling_params) -> BlankRunConfig | None:
    if sampling_params is None:
        return None
    cfg = (sampling_params.extra_args or {}).get(EXTRA_ARGS_KEY)
    if not cfg:
        return None
    try:
        parsed = BlankRunConfig(
            token_id=int(cfg["token_id"]),
            k=int(cfg["k"]),
            alpha=float(cfg["alpha"]),
            cap=float(cfg["cap"]),
            abort_after=int(cfg.get("abort_after", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if parsed.token_id < 0 or parsed.k < 1 or parsed.alpha <= 0 or parsed.cap <= 0:
        return None
    if parsed.abort_after < 0:
        return None
    return parsed


class BlankRunPenalizer:
    """Per-request consecutive-blank counters + logit penalty.

    Usage from the model runner, once per decode step:
      1. apply(logits, req_ids, get_params) BEFORE sampling
      2. update(req_ids, sampled_token_ids) AFTER sampling
    prune(live_req_ids) drops state of finished requests (called lazily).
    """

    _PRUNE_EVERY = 512

    def __init__(self):
        # req_id -> parsed config (None = request opted out; cached to avoid
        # re-parsing extra_args every step)
        self._cfgs: dict[str, BlankRunConfig | None] = {}
        # req_id -> consecutive sampled-blank count
        self._runs: dict[str, int] = {}
        # abort state: req_id -> consecutive broken-run count, req_id ->
        # real tokens since the last break (present only while counting).
        self._break_streaks: dict[str, int] = {}
        self._tokens_since_break: dict[str, int] = {}
        self._pending_aborts: list[str] = []
        self._steps = 0

    def _cfg(self, req_id: str, get_params) -> BlankRunConfig | None:
        try:
            return self._cfgs[req_id]
        except KeyError:
            cfg = parse_config(get_params(req_id))
            self._cfgs[req_id] = cfg
            if cfg is not None:
                # One line per streaming session: proves the wiring in prod logs.
                logger.info(
                    "blank-run penalty armed for %s: k=%d alpha=%.2f cap=%.1f"
                    " abort_after=%d",
                    req_id, cfg.k, cfg.alpha, cfg.cap, cfg.abort_after,
                )
            return cfg

    def apply(self, logits: torch.Tensor | None, req_ids, get_params) -> bool:
        """Subtract the penalty in place for locked requests. Returns True if
        any request in the batch has the penalty configured (callers may use
        it to skip `update` entirely for non-realtime workloads)."""
        if logits is None:
            return False
        any_active = False
        for i, req_id in enumerate(req_ids):
            cfg = self._cfg(req_id, get_params)
            if cfg is None:
                continue
            any_active = True
            run = self._runs.get(req_id, 0)
            if run > cfg.k:
                logits[i, cfg.token_id] -= min(cfg.cap, cfg.alpha * (run - cfg.k))
        return any_active

    def update(self, req_ids, sampled_token_ids) -> None:
        """Advance per-request counters from this step's sampled tokens.

        sampled_token_ids: list[int] aligned with req_ids (one token per
        request; the realtime decode path emits exactly one).
        """
        for i, req_id in enumerate(req_ids):
            cfg = self._cfgs.get(req_id)
            if cfg is None:
                continue
            if i < len(sampled_token_ids) and sampled_token_ids[i] == cfg.token_id:
                run = self._runs.get(req_id, 0) + 1
                self._runs[req_id] = run
                if run % 50 == 0:
                    logger.debug(
                        "blank run for %s: %d consecutive (k=%d)",
                        req_id, run, cfg.k,
                    )
            else:
                prev_run = self._runs.get(req_id, 0)
                if prev_run > cfg.k:
                    self._on_break(req_id, cfg, prev_run)
                elif req_id in self._tokens_since_break:
                    self._tokens_since_break[req_id] += 1
                    if self._tokens_since_break[req_id] > ABORT_MAX_TOKENS_BETWEEN:
                        # speech resumed: stop counting
                        del self._tokens_since_break[req_id]
                        self._break_streaks.pop(req_id, None)
                self._runs[req_id] = 0
        self._steps += 1
        if self._steps % self._PRUNE_EVERY == 0 and len(self._cfgs) > 2 * len(req_ids):
            self.prune(set(req_ids))

    def _on_break(self, req_id: str, cfg: BlankRunConfig, run: int) -> None:
        """A blank run > k just broke: count it, abort a stuck session."""
        since = self._tokens_since_break.get(req_id)
        if since is not None and since <= ABORT_MAX_TOKENS_BETWEEN:
            streak = self._break_streaks.get(req_id, 0) + 1
        else:
            streak = 1
        self._break_streaks[req_id] = streak
        self._tokens_since_break[req_id] = 0
        logger.info(
            "blank run broken for %s: run=%d streak=%d/%s",
            req_id, run, streak,
            cfg.abort_after if cfg.abort_after > 0 else "-",
        )
        if cfg.abort_after > 0 and streak >= cfg.abort_after:
            logger.warning(
                "blank-run abort for %s: %d broken runs in a row",
                req_id, streak,
            )
            self._pending_aborts.append(req_id)
            self._break_streaks[req_id] = 0
            del self._tokens_since_break[req_id]

    def drain_aborts(self) -> list[str]:
        if not self._pending_aborts:
            return []
        out = self._pending_aborts
        self._pending_aborts = []
        return out

    def prune(self, live_req_ids: set) -> None:
        self._cfgs = {r: c for r, c in self._cfgs.items() if r in live_req_ids}
        self._runs = {r: n for r, n in self._runs.items() if r in live_req_ids}
        self._break_streaks = {
            r: n for r, n in self._break_streaks.items() if r in live_req_ids
        }
        self._tokens_since_break = {
            r: n for r, n in self._tokens_since_break.items() if r in live_req_ids
        }
