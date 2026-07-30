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
    """Request body for the LLM test endpoint.

    Attributes:
        prompt: The user prompt. ``None`` lets the endpoint use its default.
    """

    prompt: str | None = None


class ToolResult(BaseModel):
    """A single step captured from a pipeline run.

    Attributes:
        tool: The name of the tool or step (e.g. "diagnose", "get_flag").
        args: The arguments passed to the step.
        result: The value the step returned.
    """

    tool: str
    args: dict[str, Any]
    result: str


class TestResponse(BaseModel):
    """Response for the LLM test endpoint.

    Attributes:
        model: The LiteLLM alias that produced the reply.
        prompt: The prompt actually sent (after default resolution).
        response: The model's reply text.
        elapsed_s: Wall-clock seconds for the call (model cold starts can be slow).
    """

    model: str
    prompt: str
    response: str
    elapsed_s: float


class CodeRequest(BaseModel):
    """Request body for the coding pipeline endpoint.

    Attributes:
        text: Clinical text or patient description to code.
        medical_model: LiteLLM alias of the medical model that generates the
            diagnoses (default ``ii-medical-q8``).
        k: Maximum ICD-10 codes to return per diagnosis (default 3).
    """

    text: str
    medical_model: str = "ii-medical-q8"
    k: int = 3


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
        diagnoses: The diagnoses produced by the medical model (1-10 entries).
        codes: Matching billable ICD-10-CM codes (most similar first).
        reasoning: The model's clinical reasoning for the diagnoses
            (empty string if the model didn't provide any).
        tool_results: Step-by-step trace of the pipeline (diagnose + search
            invocations) for debugging and transparency.
        elapsed_s: Wall-clock seconds for the entire pipeline.
    """

    diagnoses: list[str]
    codes: list[ICD10Match]
    reasoning: str = ""
    tool_results: list[ToolResult] | None = None
    elapsed_s: float


class DiagnoseRequest(BaseModel):
    """Request body for the dual-diagnosis endpoint.

    Attributes:
        text: Clinical text or patient description to diagnose.
        k: Maximum ICD-10 codes to return per diagnosis (default 3).
        enable_critic: Whether to run the debate-critic reconciliation loop
            when the two models disagree (default True).
    """

    text: str
    k: int = 3
    enable_critic: bool = True


class DiagnosisResult(BaseModel):
    """A single model's diagnoses, reasoning, and matching ICD-10 codes.

    Attributes:
        model: The LiteLLM alias that produced the diagnoses.
        diagnoses: The diagnoses produced by the model (1-10 entries).
        codes: Matching billable ICD-10-CM codes (most similar first).
        reasoning: The model's clinical reasoning (empty string if the
            model didn't provide any).
    """

    model: str
    diagnoses: list[str]
    codes: list[ICD10Match]
    reasoning: str = ""


class CriticRound(BaseModel):
    """One round of the debate-critic reconciliation loop.

    Attributes:
        round: Zero-indexed round number.
        reasoning: The critic's clinical reasoning for this round.
        diagnoses: The critic's reconciled diagnoses (1-10 entries).
        queries: Free-form search terms the critic suggested for finding
            ICD-10 codes (used alongside diagnoses for vector search).
        codes: ICD-10 matches found for this round's diagnoses + queries.
        done: Whether the critic signalled confidence (``true`` ends the loop).
    """

    round: int
    reasoning: str = ""
    diagnoses: list[str]
    queries: list[str] = []
    codes: list[ICD10Match] = []
    done: bool = False


class DualDiagnoseResponse(BaseModel):
    """Response from the dual-diagnosis pipeline.

    Attributes:
        results: One :class:`DiagnosisResult` per medical model, in the order
            the models are configured (ii-medical-q8, deepseek-r1-medical-cot).
        critic_triggered: Whether the debate-critic loop ran (models disagreed).
        critic_rounds: Full trace of each critic reconciliation round (empty
            if the critic was not triggered or disabled).
        elapsed_s: Wall-clock seconds for the entire pipeline.
    """

    results: list[DiagnosisResult]
    critic_triggered: bool = False
    critic_rounds: list[CriticRound] = []
    elapsed_s: float
