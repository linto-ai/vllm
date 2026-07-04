# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for BlankRunPenaltyLogitsProcessor."""

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BlankRunPenaltyLogitsProcessor
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

BLANK = 32
VOCAB = 128
K = 10
ALPHA = 0.5
CAP = 4.0


def make_params(k=K, alpha=ALPHA, cap=CAP, token_id=BLANK):
    return SamplingParams(
        extra_args={
            "blank_run_penalty": {
                "token_id": token_id,
                "k": k,
                "alpha": alpha,
                "cap": cap,
            }
        }
    )


def make_proc():
    return BlankRunPenaltyLogitsProcessor(None, torch.device("cpu"), False)


def add(proc, index, params, out_ids):
    proc.update_state(
        BatchUpdate(
            batch_size=index + 1,
            removed=[],
            added=[(index, params, [1, 2], out_ids)],
            moved=[],
        )
    )


def apply_ones(proc, rows=1):
    logits = torch.ones(rows, VOCAB)
    return proc.apply(logits)


def test_disabled_without_extra_args():
    proc = make_proc()
    add(proc, 0, SamplingParams(), [BLANK] * 50)
    out = apply_ones(proc)
    assert torch.all(out == 1.0)


def test_no_penalty_at_or_below_k():
    proc = make_proc()
    out_ids: list[int] = [BLANK] * K
    add(proc, 0, make_params(), out_ids)
    out = apply_ones(proc)
    assert out[0, BLANK] == 1.0


def test_progressive_penalty_above_k():
    proc = make_proc()
    out_ids = [7] + [BLANK] * (K + 3)
    add(proc, 0, make_params(), out_ids)
    out = apply_ones(proc)
    assert out[0, BLANK] == pytest.approx(1.0 - ALPHA * 3)
    # only the blank column of that row is touched
    assert out[0, BLANK - 1] == 1.0
    assert torch.all(out[0, :BLANK] == 1.0)


def test_cap_saturation():
    proc = make_proc()
    out_ids = [BLANK] * 500
    add(proc, 0, make_params(), out_ids)
    out = apply_ones(proc)
    assert out[0, BLANK] == pytest.approx(1.0 - CAP)


def test_reset_on_non_blank_via_live_reference():
    proc = make_proc()
    out_ids = [BLANK] * (K + 8)
    add(proc, 0, make_params(), out_ids)
    assert apply_ones(proc)[0, BLANK] < 1.0
    # the processor holds a live reference: appending a speech token resets
    out_ids.append(99)
    assert apply_ones(proc)[0, BLANK] == 1.0
    # and a fresh run below K stays untouched
    out_ids.extend([BLANK] * K)
    assert apply_ones(proc)[0, BLANK] == 1.0
    out_ids.append(BLANK)
    assert apply_ones(proc)[0, BLANK] == pytest.approx(1.0 - ALPHA)


def test_removed_request_cleans_state():
    proc = make_proc()
    add(proc, 0, make_params(), [BLANK] * 100)
    proc.update_state(
        BatchUpdate(batch_size=0, removed=[0], added=[], moved=[])
    )
    assert apply_ones(proc)[0, BLANK] == 1.0


def test_moved_request_follows_index():
    proc = make_proc()
    add(proc, 0, make_params(), [BLANK] * 100)
    proc.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[],
            moved=[(0, 1, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    out = apply_ones(proc, rows=2)
    assert out[0, BLANK] == 1.0
    assert out[1, BLANK] == pytest.approx(1.0 - CAP)


def test_index_beyond_batch_is_ignored():
    proc = make_proc()
    add(proc, 3, make_params(), [BLANK] * 100)
    out = apply_ones(proc, rows=1)  # smaller logits batch than index
    assert torch.all(out == 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"token_id": 32},  # missing fields
        {"token_id": -1, "k": 10, "alpha": 0.5, "cap": 4.0},
        {"token_id": 32, "k": 0, "alpha": 0.5, "cap": 4.0},
        {"token_id": 32, "k": 10, "alpha": 0.0, "cap": 4.0},
        {"token_id": 32, "k": 10, "alpha": 0.5, "cap": -1.0},
        "not-a-dict",
    ],
)
def test_validate_params_rejects_bad_config(bad):
    with pytest.raises(ValueError):
        BlankRunPenaltyLogitsProcessor.validate_params(
            SamplingParams(extra_args={"blank_run_penalty": bad})
        )


def test_validate_params_accepts_absent_and_valid():
    BlankRunPenaltyLogitsProcessor.validate_params(SamplingParams())
    BlankRunPenaltyLogitsProcessor.validate_params(make_params())
