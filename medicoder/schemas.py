"""Pydantic models shared across the API."""

from __future__ import annotations

from pydantic import BaseModel


class ICD10Code(BaseModel):
    order_number: int
    code: str
    code_type: int
    is_billable: bool
    short_desc: str
    long_desc: str


class TestRequest(BaseModel):
    prompt: str | None = None


class TestResponse(BaseModel):
    model: str
    prompt: str
    response: str
    elapsed_s: float
