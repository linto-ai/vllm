# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the worker-side BlankRunPenalizer."""

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.worker.blank_run_penalty import BlankRunPenalizer, parse_config

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


def run_steps(pen, req_params, tokens_per_step):
    """Drive apply+update over a token schedule.

    req_params: {req_id: SamplingParams}
    tokens_per_step: list of {req_id: sampled_token}
    Returns the logits of the LAST apply (ones baseline).
    """
    get_params = lambda rid: req_params.get(rid)
    logits = None
    for step in tokens_per_step:
        req_ids = list(step.keys())
        logits = torch.ones(len(req_ids), VOCAB)
        pen.apply(logits, req_ids, get_params)
        pen.update(req_ids, [step[r] for r in req_ids])
    return logits


def test_opt_out_request_untouched():
    pen = BlankRunPenalizer()
    logits = run_steps(pen, {"a": SamplingParams()}, [{"a": BLANK}] * 50)
    assert torch.all(logits == 1.0)


def test_no_penalty_at_or_below_k():
    pen = BlankRunPenalizer()
    params = {"a": make_params()}
    # after K blanks the (K+1)th apply sees run == K -> no penalty yet
    logits = run_steps(pen, params, [{"a": BLANK}] * (K + 1))
    assert logits[0, BLANK] == 1.0


def test_progressive_penalty_and_cap():
    pen = BlankRunPenalizer()
    params = {"a": make_params()}
    run_steps(pen, params, [{"a": BLANK}] * (K + 3))
    logits = torch.ones(1, VOCAB)
    pen.apply(logits, ["a"], lambda r: params[r])
    assert logits[0, BLANK] == pytest.approx(1.0 - ALPHA * 3)
    assert torch.all(logits[0, :BLANK] == 1.0)
    run_steps(pen, params, [{"a": BLANK}] * 500)
    logits = torch.ones(1, VOCAB)
    pen.apply(logits, ["a"], lambda r: params[r])
    assert logits[0, BLANK] == pytest.approx(1.0 - CAP)


def test_reset_on_non_blank():
    pen = BlankRunPenalizer()
    params = {"a": make_params()}
    run_steps(pen, params, [{"a": BLANK}] * (K + 8))
    run_steps(pen, params, [{"a": 99}])
    logits = torch.ones(1, VOCAB)
    pen.apply(logits, ["a"], lambda r: params[r])
    assert logits[0, BLANK] == 1.0


def test_counter_survives_request_reentry():
    """The realtime path re-adds the request per chunk; the counter is keyed
    by req_id and must keep accumulating across re-entries."""
    pen = BlankRunPenalizer()
    params = {"a": make_params()}
    for _ in range(K + 5):  # one 1-token "segment" per loop turn
        run_steps(pen, params, [{"a": BLANK}])
    logits = torch.ones(1, VOCAB)
    pen.apply(logits, ["a"], lambda r: params[r])
    assert logits[0, BLANK] == pytest.approx(1.0 - ALPHA * 5)


def test_per_request_isolation():
    pen = BlankRunPenalizer()
    params = {"a": make_params(), "b": make_params()}
    steps = [{"a": BLANK, "b": 7}] * (K + 6)
    run_steps(pen, params, steps)
    logits = torch.ones(2, VOCAB)
    pen.apply(logits, ["a", "b"], lambda r: params[r])
    assert logits[0, BLANK] == pytest.approx(1.0 - ALPHA * 6)
    assert logits[1, BLANK] == 1.0


def test_prune_drops_dead_requests():
    pen = BlankRunPenalizer()
    params = {"a": make_params()}
    run_steps(pen, params, [{"a": BLANK}] * (K + 5))
    pen.prune(set())
    logits = torch.ones(1, VOCAB)
    pen.apply(logits, ["a"], lambda r: params[r])
    assert logits[0, BLANK] == 1.0  # counter gone, run restarts


@pytest.mark.parametrize(
    "bad",
    [
        {"token_id": 32},
        {"token_id": -1, "k": 10, "alpha": 0.5, "cap": 4.0},
        {"token_id": 32, "k": 0, "alpha": 0.5, "cap": 4.0},
        {"token_id": 32, "k": 10, "alpha": 0.0, "cap": 4.0},
        {"token_id": 32, "k": 10, "alpha": 0.5, "cap": -1.0},
        "not-a-dict",
    ],
)
def test_parse_config_rejects_bad(bad):
    assert parse_config(SamplingParams(extra_args={"blank_run_penalty": bad})) is None


def test_parse_config_absent_and_valid():
    assert parse_config(SamplingParams()) is None
    assert parse_config(make_params()) is not None


# ---------------------------------------------------------------------------
# Session abort (abort_after > 0)
# ---------------------------------------------------------------------------

def make_abort_params(abort_after=2, **kw):
    params = make_params(**kw)
    params.extra_args["blank_run_penalty"]["abort_after"] = abort_after
    return params


def test_parse_config_abort_field():
    assert parse_config(make_abort_params(abort_after=2)).abort_after == 2
    assert parse_config(make_params()).abort_after == 0  # default: off


def test_two_broken_runs_abort():
    # stuck session: >K blanks, one forced token, again -> 2 breaks in a row
    pen = BlankRunPenalizer()
    params = {"a": make_abort_params()}
    schedule = (
        [{"a": BLANK}] * (K + 5) + [{"a": 7}]
        + [{"a": BLANK}] * (K + 5) + [{"a": 7}]
    )
    run_steps(pen, params, schedule)
    assert pen.drain_aborts() == ["a"]
    assert pen.drain_aborts() == []


def test_true_silence_never_aborts():
    # margin > cap: the run never breaks, so no abort can fire
    pen = BlankRunPenalizer()
    params = {"a": make_abort_params()}
    run_steps(pen, params, [{"a": BLANK}] * (10 * K))
    assert pen.drain_aborts() == []


def test_speech_resumption_resets_streak():
    # long pause -> one break -> real speech disarms; a later pause is streak=1
    pen = BlankRunPenalizer()
    params = {"a": make_abort_params()}
    schedule = (
        [{"a": BLANK}] * (K + 5) + [{"a": 7}]
        + [{"a": 8}] * 10
        + [{"a": BLANK}] * (K + 5) + [{"a": 9}]
    )
    run_steps(pen, params, schedule)
    assert pen.drain_aborts() == []


def test_abort_disabled_by_default():
    pen = BlankRunPenalizer()
    params = {"a": make_params()}
    schedule = ([{"a": BLANK}] * (K + 5) + [{"a": 7}]) * 4
    run_steps(pen, params, schedule)
    assert pen.drain_aborts() == []


def test_few_tokens_between_breaks_keep_streak():
    # a stuck session can force 2-3 junk tokens per break, not always one
    pen = BlankRunPenalizer()
    params = {"a": make_abort_params()}
    schedule = (
        [{"a": BLANK}] * (K + 5) + [{"a": 7}, {"a": 7}]
        + [{"a": BLANK}] * (K + 5) + [{"a": 7}]
    )
    run_steps(pen, params, schedule)
    assert pen.drain_aborts() == ["a"]


def test_prune_clears_abort_state():
    pen = BlankRunPenalizer()
    params = {"a": make_abort_params()}
    run_steps(pen, params, [{"a": BLANK}] * (K + 5) + [{"a": 7}])
    pen.prune(set())
    assert pen._break_streaks == {}
    assert pen._tokens_since_break == {}
