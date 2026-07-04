# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the lazy-arming handshake of RealtimeConnection.

A non-final commit received before any audio must NOT start a generation:
the engine would run a request over an empty audio queue, and a client that
disconnects before sending audio would leave a dead-born session behind
(padding-only input, zero-embedding fallback). Generation start is deferred
to the first append instead. No GPU or engine is required here: handle_event
is driven directly with fakes.
"""

import json

import numpy as np
import pybase64 as base64
import pytest

from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection

MODEL = "test-realtime-model"


class FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, data: str):
        self.sent.append(json.loads(data))


class FakeServing:
    def _is_model_supported(self, model):
        return model == MODEL


def pcm_chunk(n_samples: int = 1600) -> str:
    audio = (np.zeros(n_samples)).astype(np.int16).tobytes()
    return base64.b64encode(audio).decode("utf-8")


@pytest.fixture
def conn(monkeypatch):
    connection = RealtimeConnection(FakeWebSocket(), FakeServing())
    connection.generation_starts = 0

    async def fake_start_generation():
        connection.generation_starts += 1

    monkeypatch.setattr(connection, "start_generation", fake_start_generation)
    return connection


def sent_types(conn):
    return [e["type"] for e in conn.websocket.sent]


@pytest.mark.asyncio
async def test_session_update_is_acked(conn):
    await conn.handle_event({"type": "session.update", "model": MODEL})
    assert sent_types(conn) == ["session.updated"]
    assert conn._is_model_validated


@pytest.mark.asyncio
async def test_commit_before_audio_defers_generation(conn):
    await conn.handle_event({"type": "session.update", "model": MODEL})
    await conn.handle_event({"type": "input_audio_buffer.commit"})

    assert conn.generation_starts == 0
    assert conn._arm_pending

    await conn.handle_event(
        {"type": "input_audio_buffer.append", "audio": pcm_chunk()}
    )

    assert conn.generation_starts == 1
    assert not conn._arm_pending
    # Audio chunk still queued for the generation to consume
    assert conn.audio_queue.qsize() == 1


@pytest.mark.asyncio
async def test_commit_after_audio_starts_immediately(conn):
    await conn.handle_event({"type": "session.update", "model": MODEL})
    await conn.handle_event(
        {"type": "input_audio_buffer.append", "audio": pcm_chunk()}
    )
    assert conn.generation_starts == 0  # append alone must not start anything

    await conn.handle_event({"type": "input_audio_buffer.commit"})
    assert conn.generation_starts == 1
    assert not conn._arm_pending


@pytest.mark.asyncio
async def test_final_commit_on_never_fed_session_sends_empty_done(conn):
    """Armed then ended without audio: the client must not wait forever."""
    await conn.handle_event({"type": "session.update", "model": MODEL})
    await conn.handle_event({"type": "input_audio_buffer.commit"})
    await conn.handle_event({"type": "input_audio_buffer.commit", "final": True})

    assert conn.generation_starts == 0
    done = [e for e in conn.websocket.sent if e["type"] == "transcription.done"]
    assert len(done) == 1
    assert done[0]["text"] == ""
    # Nothing queued: no sentinel for a generation that never existed
    assert conn.audio_queue.qsize() == 0
    assert not conn._arm_pending


@pytest.mark.asyncio
async def test_final_commit_with_audio_queues_sentinel(conn):
    await conn.handle_event({"type": "session.update", "model": MODEL})
    await conn.handle_event({"type": "input_audio_buffer.commit"})
    await conn.handle_event(
        {"type": "input_audio_buffer.append", "audio": pcm_chunk()}
    )
    await conn.handle_event({"type": "input_audio_buffer.commit", "final": True})

    assert conn.generation_starts == 1
    # audio chunk + None sentinel
    assert conn.audio_queue.qsize() == 2


@pytest.mark.asyncio
async def test_second_utterance_commit_starts_immediately(conn):
    """Multi-utterance flow: once audio has ever been received, a later
    non-final commit starts generation immediately (previous behavior)."""
    await conn.handle_event({"type": "session.update", "model": MODEL})
    await conn.handle_event({"type": "input_audio_buffer.commit"})
    await conn.handle_event(
        {"type": "input_audio_buffer.append", "audio": pcm_chunk()}
    )
    assert conn.generation_starts == 1

    # Simulate end of first utterance, then a new commit
    await conn.handle_event({"type": "input_audio_buffer.commit", "final": True})
    conn.generation_task = None
    await conn.handle_event({"type": "input_audio_buffer.commit"})
    assert conn.generation_starts == 2


@pytest.mark.asyncio
async def test_commit_without_session_update_still_rejected(conn):
    await conn.handle_event({"type": "input_audio_buffer.commit"})
    errors = [e for e in conn.websocket.sent if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "model_not_validated"
    assert not conn._arm_pending
    assert conn.generation_starts == 0
