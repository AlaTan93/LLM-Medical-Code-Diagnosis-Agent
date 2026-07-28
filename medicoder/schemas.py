"""Pydantic models shared across the API."""

from __future__ import annotations

from typing import Any

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


class ToolResult(BaseModel):
    """A single step captured from a pipeline or agent run.

    Attributes:
        tool: The name of the tool or step (e.g. "diagnose", "get_flag").
        args: The arguments passed to the step.
        result: The value the step returned.
    """

    tool: str
    args: dict[str, Any]
    result: str


class TestResponse(BaseModel):
    """Response for the LLM test and agent endpoints.

    Attributes:
        model: The LiteLLM alias that produced the reply.
        prompt: The prompt actually sent (after default resolution).
        response: The model/agent's reply text.
        elapsed_s: Wall-clock seconds for the call (model cold starts can be slow).
        tool_results: Tool invocations captured during an agent run (``None`` for
            the plain ``/test`` route). Each entry records the tool name, the
            arguments the model supplied, and the value the tool returned — the
            ground-truth output, independent of the model's textual summary.
    """

    model: str
    prompt: str
    response: str
    elapsed_s: float
    tool_results: list[ToolResult] | None = None


class CodeRequest(BaseModel):
    """Request body for the coding pipeline endpoint.

    Attributes:
        text: Clinical text or patient description to code.
        medical_model: LiteLLM alias of the medical model that generates the
            one-sentence diagnosis (default ``ii-medical-q8``).
    """

    text: str
    medical_model: str = "ii-medical-q8"


class ICD10Match(BaseModel):
    """An ICD-10-CM code matched by vector similarity.

    Attributes:
        code: The ICD-10-CM code string (e.g. "E11.9").
        short_desc: Short human-readable description.
        long_desc: Full human-readable description.
        similarity: Cosine similarity (0-1) between the query and the code's
            embedding; higher is a better match.
    """

    code: str
    short_desc: str
    long_desc: str
    similarity: float


class CodeResponse(BaseModel):
    """Response from the coding pipeline.

    Attributes:
        diagnosis: The one-sentence diagnosis produced by the medical model.
        codes: Top matching billable ICD-10-CM codes (most similar first).
        tool_results: Step-by-step trace (diagnose + search_icd10 invocations).
        elapsed_s: Wall-clock seconds for the entire pipeline.
    """

    diagnosis: str
    codes: list[ICD10Match]
    tool_results: list[ToolResult] | None = None
    elapsed_s: float
