# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blank-run penalty: break self-sustained runs of a designated blank token.

Streaming speech-to-text models that re-ingest their own output can fall into
a self-sustained decoding rut: once the context window fills with the
blank/silence token, greedy decoding keeps emitting it over real speech for
minutes (observed with Voxtral realtime; same family as Whisper repetition
loops). Temperature does not help: the distribution collapses to P(blank)~1.

This processor applies a progressive penalty to the blank token once a
request has emitted it more than K consecutive times:

    penalty = min(cap, alpha * (run_length - K))

Healthy blank runs (inter-sentence silences, tens of frames) stay below K and
are never touched. Real extended silence is capped at `cap`, chosen so that
genuinely silent audio (blank margin >> cap) keeps decoding as silence while
a marginal rut over speech (small margin) is broken.

Enabled per request via::

    SamplingParams(extra_args={"blank_run_penalty": {
        "token_id": 32, "k": 200, "alpha": 0.5, "cap": 7.0}})
"""

from dataclasses import dataclass

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

EXTRA_ARGS_KEY = "blank_run_penalty"


@dataclass(frozen=True)
class _BlankRunConfig:
    token_id: int
    k: int
    alpha: float
    cap: float


def _parse_config(params: SamplingParams) -> _BlankRunConfig | None:
    cfg = (params.extra_args or {}).get(EXTRA_ARGS_KEY)
    if not cfg:
        return None
    return _BlankRunConfig(
        token_id=int(cfg["token_id"]),
        k=int(cfg["k"]),
        alpha=float(cfg["alpha"]),
        cap=float(cfg["cap"]),
    )


class BlankRunPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # index -> (config, live reference to the request's output token ids)
        self.req_info: dict[int, tuple[_BlankRunConfig, list[int]]] = {}

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        cfg = (sampling_params.extra_args or {}).get(EXTRA_ARGS_KEY)
        if cfg is None:
            return
        if not isinstance(cfg, dict):
            raise ValueError(f"{EXTRA_ARGS_KEY} must be a dict")
        try:
            parsed = _parse_config(sampling_params)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"{EXTRA_ARGS_KEY} requires numeric fields "
                f"token_id, k, alpha, cap: {e}"
            ) from e
        assert parsed is not None
        if parsed.token_id < 0:
            raise ValueError(f"{EXTRA_ARGS_KEY}.token_id must be >= 0")
        if parsed.k < 1:
            raise ValueError(f"{EXTRA_ARGS_KEY}.k must be >= 1")
        if parsed.alpha <= 0 or parsed.cap <= 0:
            raise ValueError(f"{EXTRA_ARGS_KEY}.alpha and .cap must be > 0")

    def is_argmax_invariant(self) -> bool:
        # The whole point is to flip greedy picks out of a blank rut.
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        def new_state(params: SamplingParams, prompt_ids, output_tok_ids):
            cfg = _parse_config(params)
            if cfg is None or cfg.k < 1:
                return None
            return (cfg, output_tok_ids)

        process_dict_updates(self.req_info, batch_update, new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits
        num_rows = logits.shape[0]
        for index, (cfg, out_ids) in self.req_info.items():
            if index >= num_rows:
                continue
            # Count the trailing run of blank tokens. The penalty saturates at
            # `cap`, so never scan deeper than the saturation length.
            max_scan = cfg.k + int(cfg.cap / cfg.alpha) + 1
            run = 0
            for tok in reversed(out_ids):
                if tok != cfg.token_id:
                    break
                run += 1
                if run >= max_scan:
                    break
            if run > cfg.k:
                penalty = min(cfg.cap, cfg.alpha * (run - cfg.k))
                logits[index, cfg.token_id] -= penalty
        return logits
