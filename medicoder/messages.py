"""Helpers for extracting data from LangGraph/LangChain message lists.

The ``/agent`` route needs to pull structured data out of the message list
returned by ``agent.invoke``.  Centralising that logic here keeps it out of
the route module.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from medicoder.schemas import ToolResult


def extract_tool_results(messages: list) -> list[ToolResult]:
    """Collect tool invocations from a LangGraph message list.

    Walks the message history pairing each ``AIMessage`` tool-call (which
    carries the tool name + arguments) with its matching ``ToolMessage`` (which
    carries the tool's return value) via ``tool_call_id``.  This captures the
    ground-truth tool output regardless of how the model summarises (or fails to
    summarise) it in its final text.

    Args:
        messages: The ``messages`` list from an agent ``invoke`` result.

    Returns:
        One :class:`ToolResult` per executed tool call, in execution order.
    """
    pending: dict[str, tuple[str, dict]] = {}
    results: list[ToolResult] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                if tc_id is not None:
                    pending[tc_id] = (name, args)
        elif isinstance(msg, ToolMessage):
            tc_id = msg.tool_call_id
            name, args = pending.get(tc_id, (msg.name or "", {}))
            content = msg.content
            if not isinstance(content, str):
                content = str(content)
            results.append(ToolResult(tool=name, args=args or {}, result=content))
    return results


def last_content(messages: list) -> str:
    """Extract the text content of the last message in a list.

    Chat models may return content as a string or as a list of content blocks
    (e.g. ``[{"type": "text", "text": "…"}]``); this collapses either form to a
    plain string.

    Args:
        messages: The messages list from an agent ``invoke`` result.

    Returns:
        The text content of the final message, or ``""`` if the list is empty.
    """
    if not messages:
        return ""
    raw = getattr(messages[-1], "content", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(raw)
