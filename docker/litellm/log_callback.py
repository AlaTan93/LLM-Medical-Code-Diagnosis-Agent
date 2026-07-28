"""Custom LiteLLM success callback.

Logs every LLM call — prompt, thinking, output, and tool calls — to the
Postgres ``llm_call_log`` table. Registered in ``docker/litellm/config.yaml``
via ``litellm_settings.callbacks``. Writes go through LiteLLM's own Prisma
client (``DATABASE_URL`` → the ``litellm`` DB), so no extra Postgres driver is
needed in the litellm image.

The prompt (request messages) lives in ``messages``; the model's reasoning in
``thinking`` (``reasoning_content`` / ``<think>``); the reply text in
``output``; any tool invocations in ``tool_calls``. Tool *results* are captured
on the next call, where they appear as ``tool``-role entries in ``messages``.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

_CREATE_SQL = """CREATE TABLE IF NOT EXISTS llm_call_log (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_id TEXT,
  model TEXT,
  call_type TEXT,
  messages JSONB,
  thinking TEXT,
  output TEXT,
  tool_calls JSONB,
  prompt_tokens INT,
  completion_tokens INT,
  total_tokens INT,
  latency_ms INT
)"""

_INSERT_SQL = (
    "INSERT INTO llm_call_log "
    "(request_id, model, call_type, messages, thinking, output, tool_calls, "
    "prompt_tokens, completion_tokens, total_tokens, latency_ms) "
    "VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::jsonb, $8, $9, $10, $11)"
)


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps(repr(obj), ensure_ascii=False)


class LLMCallLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._table_ready = False

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        try:
            await self._ensure_table()

            slo = kwargs.get("standard_logging_object") or {}
            model = kwargs.get("model") or slo.get("model", "")
            call_type = slo.get("call_type", "")
            messages = slo.get("messages") or kwargs.get("messages") or []
            request_id = slo.get("id") or kwargs.get("litellm_call_id") or ""

            thinking = output = ""
            tool_calls = None
            choices = getattr(response_obj, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    output = getattr(msg, "content", "") or ""
                    thinking = (
                        getattr(msg, "reasoning_content", None)
                        or getattr(msg, "thinking", None)
                        or ""
                    )
                    tc = getattr(msg, "tool_calls", None)
                    if tc:
                        tool_calls = _json(tc)

            pt = ct = tt = 0
            usage = getattr(response_obj, "usage", None) or slo.get("usage") or {}
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens", 0) or 0
                ct = usage.get("completion_tokens", 0) or 0
                tt = usage.get("total_tokens", 0) or 0
            else:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                tt = getattr(usage, "total_tokens", 0) or 0

            latency_ms = None
            if start_time and end_time:
                try:
                    latency_ms = int((end_time - start_time).total_seconds() * 1000)
                except Exception:
                    latency_ms = None

            await self._exec(
                _INSERT_SQL,
                request_id,
                model,
                call_type,
                _json(messages),
                thinking,
                output,
                tool_calls,
                pt,
                ct,
                tt,
                latency_ms,
            )
        except Exception as e:
            print(f"[llm_call_log] ERROR {e!r}", flush=True)
            traceback.print_exc()

    async def _ensure_table(self) -> None:
        if self._table_ready:
            return
        await self._exec(_CREATE_SQL)
        self._table_ready = True

    async def _exec(self, sql: str, *params: Any) -> None:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return
        db = prisma_client.db
        try:
            await db.execute_raw(sql, *params)
        except Exception:
            await db.query_raw(sql, *params)


llm_call_logger = LLMCallLogger()
