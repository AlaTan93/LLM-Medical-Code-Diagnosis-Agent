"""Pydantic models shared across the API."""

from __future__ import annotations

from pydantic import BaseModel


class ICD10Code(BaseModel):
    """A row from the ICD-10-CM order table.

    Attributes:
        order_number: CMS-assigned sequential order number (primary key).
        code: The ICD-10-CM code string (e.g. "E11.9").
        code_type: 0 for a category header, 1 for a billable code.
        is_billable: Whether the code is billable (derived from code_type).
        short_desc: Short human-readable description.
        long_desc: Full human-readable description.
    """

    order_number: int
    code: str
    code_type: int
    is_billable: bool
    short_desc: str
    long_desc: str


class TestRequest(BaseModel):
    """Request body for the LLM test and agent endpoints.

    Attributes:
        prompt: The user prompt. ``None`` lets the endpoint use its default.
    """

    prompt: str | None = None


class TestResponse(BaseModel):
    """Response for the LLM test and agent endpoints.

    Attributes:
        model: The LiteLLM alias that produced the reply.
        prompt: The prompt actually sent (after default resolution).
        response: The model/agent's reply text.
        elapsed_s: Wall-clock seconds for the call (model cold starts can be slow).
    """

    model: str
    prompt: str
    response: str
    elapsed_s: float
