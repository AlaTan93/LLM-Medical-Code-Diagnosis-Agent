"""Shared medical pipeline functions used by ``/code`` and ``/diagnose``.

- :func:`diagnose` — forwards clinical text to a medical model and returns
  1-10 independent diagnoses plus the model's clinical reasoning.
- :func:`search_icd10` — batch-embeds multiple diagnosis queries and runs a
  hybrid vector-primary + FTS-boost search over billable ICD-10-CM codes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError, field_validator

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.schemas import ICD10Match

_MAX_DIAGNOSES = int(os.environ.get("MAX_DIAGNOSES", "10"))

_MAX_RETRIES = int(os.environ.get("DIAGNOSE_MAX_RETRIES", "4"))
_TEMP_INCREMENT = float(os.environ.get("DIAGNOSE_TEMP_INCREMENT", "0.05"))
_MAX_TEMP = float(os.environ.get("DIAGNOSE_MAX_TEMP", "0.3"))

_DIAGNOSE_TOKENS = int(os.environ.get("DIAGNOSE_MAX_TOKENS", "4096"))
_DIAGNOSE_CONCISE_TOKENS = int(os.environ.get("DIAGNOSE_CONCISE_TOKENS", "2048"))

_FTS_BOOST = float(os.environ.get("FTS_BOOST", "0.0"))
_FTS_FLOOR = float(os.environ.get("FTS_FLOOR", "0.50"))
_CANDIDATE_MULT = int(os.environ.get("SEARCH_CANDIDATE_MULT", "5"))

_DIAGNOSE_SYSTEM = (
    "Analyze the patient's clinical presentation, then produce diagnoses "
    "using standard ICD-10-CM diagnostic terminology.\n"
    "Output a JSON object with two fields:\n"
    '- "reasoning": Your brief clinical reasoning (analysis of symptoms, '
    "findings, and conclusions).\n"
    '- "diagnoses": An array of 1-10 concise diagnostic phrases.\n\n'
    "ICD-10-CM naming conventions:\n"
    '- Use "Malignant neoplasm of [site]" — not "cancer" or "carcinoma"\n'
    '- Use "Unspecified" when the documentation does not specify the '
    "anatomical site or type\n"
    '- Include clinical qualifiers where documented (e.g., "acute", '
    '"in remission", "without complications")\n\n'
    "Do NOT include ICD-10 codes, code numbers, or 'patient has'.\n\n"
    "Example:\n"
    '{"reasoning": "Biopsy confirms prostate adenocarcinoma with no '
    'evidence of metastasis. Patient also has a history of opioid '
    'dependence, currently in remission.", "diagnoses": '
    '["Malignant neoplasm of prostate", "Opioid dependence, in remission"]}'
)

_DIAGNOSE_SYSTEM_CONCISE = (
    "Output a JSON object with a \"diagnoses\" array of 1-10 concise "
    "diagnostic phrases using standard ICD-10-CM terminology "
    '(e.g., "Malignant neoplasm of [site]", not "cancer").\n'
    "Use \"Unspecified\" when the site or type is not documented.\n"
    "Do NOT include ICD-10 codes, code numbers, or 'patient has'.\n"
    "Do NOT include reasoning or explanation.\n\n"
    'Example: {"diagnoses": ["Malignant neoplasm of prostate"]}'
)


_BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
_NUM_RE = re.compile(r"^[\d]+[.)]?\s*")
_BULLET_RE = re.compile(r"^[-*]\s*")


@dataclass
class ModelOutput:
    """A medical model's output bundle: diagnoses, codes, and reasoning.

    Groups the per-model quartet so that :func:`medicoder.critic.diagnose_critic`
    and :func:`medicoder.critic.build_critic_context` accept 2 positional
    args instead of 10.
    """

    model: str
    diagnoses: list[str]
    codes: list[ICD10Match]
    reasoning: str


def clean_str_list(v: list[str]) -> list[str]:
    """Strip and filter a list of strings, capping at _MAX_DIAGNOSES."""
    return [s.strip() for s in v if s.strip()][:_MAX_DIAGNOSES]


class DiagnosisOutput(BaseModel):
    """Pydantic schema for validating JSON diagnosis output from models.

    Captures both the model's clinical reasoning and its resulting
    diagnoses in a structured format.  The ``reasoning`` field makes the
    model's thought process observable — useful for debugging, the critic
    agent's context, and evaluation transparency.

    Attributes:
        reasoning: The model's clinical reasoning (empty string if the
            model didn't include it).
        diagnoses: 1-10 clean diagnosis strings.
    """

    reasoning: str = ""
    diagnoses: list[str]

    @field_validator("diagnoses", mode="after")
    @classmethod
    def clean_diagnoses(cls, v: list[str]) -> list[str]:
        return clean_str_list(v)


def _extract_json(raw: str) -> dict | None:
    """Try to extract a JSON object from raw model output.

    Models don't always return clean JSON — some wrap it in markdown
    fences, prepend prose, or embed it mid-sentence.  This function
    tries three strategies in order of decreasing strictness:

    1. Direct ``json.loads`` on the full text (best case).
    2. Extraction from a ``\\`\\`\\`json`` markdown code fence.
    3. Substring from the first ``{`` to the last ``}`` (catches JSON
       embedded after prose or thinking blocks).

    Returns ``None`` if no valid JSON object is found.
    """
    raw = raw.strip()

    # Strategy 1 — direct parse.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract from markdown code fences.
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Strategy 3 — grab the outermost { … } substring.
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(raw[first : last + 1])
        except json.JSONDecodeError:
            pass

    return None


def _parse_diagnosis_output(raw: str) -> DiagnosisOutput | None:
    """Extract and validate a ``DiagnosisOutput`` from raw model text.

    Tries :func:`_extract_json` then constructs a ``DiagnosisOutput``.
    Returns the validated instance or ``None`` if parsing or validation
    fails.
    """
    data = _extract_json(raw)
    if data is None:
        return None
    try:
        return DiagnosisOutput(**data)
    except (ValidationError, TypeError):
        return None


def _parse_diagnoses(raw: str) -> list[str]:
    """Parse raw model output into a list of clean diagnosis strings.

    Strips ``\\boxed{...}`` wrappers, leading numbers/bullets, and blank lines.
    Caps at :data:`_MAX_DIAGNOSES` entries.

    Args:
        raw: The raw text returned by the model (thinking blocks already
            separated by :func:`medicoder.proxy.extract_thinking`).

    Returns:
        A list of 0-_MAX_DIAGNOSES diagnosis strings.
    """
    text = raw.strip()

    m = _BOXED_RE.search(text)
    if m:
        text = m.group(1).strip()

    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _NUM_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines[:_MAX_DIAGNOSES]


def _try_parse_output(raw: str) -> tuple[list[str], str] | None:
    """Parse raw model output into diagnoses, trying JSON then regex.

    Tries structured JSON parsing (preferred) first, then falls back to
    line-based regex parsing for models that ignore JSON format.

    Returns ``(diagnoses, reasoning)`` on success, or ``None`` if no
    diagnoses could be extracted.
    """
    result = _parse_diagnosis_output(raw)
    if isinstance(result, DiagnosisOutput) and result.diagnoses:
        return result.diagnoses, result.reasoning

    diagnoses = _parse_diagnoses(raw)
    if diagnoses:
        return diagnoses, ""

    return None


def diagnose(
    text: str,
    model: str,
    *,
    temperature: float = 0.1,
) -> tuple[list[str], str, str]:
    """Forward clinical text to a medical model for diagnoses and reasoning.

    Uses an incremental retry strategy: the first attempt uses the full
    system prompt (with reasoning) and a 4096-token budget.  If the model
    produces zero diagnoses (e.g. thinking loop consumed all tokens or the
    output was unparseable), subsequent retries switch to a concise prompt
    that suppresses thinking, a 2048-token budget, and a slightly higher
    temperature (+0.05 per attempt, capped at 0.3).

    Pipeline per attempt: system prompt → model call → JSON extraction →
    Pydantic validation → fallback to regex parsing.

    Args:
        text: Clinical text or patient description.
        model: LiteLLM alias for the medical model (e.g. ``medgemma-27b-q4_k_s``).
        temperature: Base sampling temperature (default 0.1).  Retries
            increment this by 0.05 per attempt up to :data:`_MAX_TEMP`.

    Returns:
        A ``(diagnoses, reasoning, thinking)`` tuple where *diagnoses*
        is a list of 0-_MAX_DIAGNOSES strings, *reasoning* is the
        model's clinical explanation (empty string on fallback or
        concise-prompt retries), and *thinking* is the raw
        ``<think>`` block content (empty for non-reasoning models).
    """
    for attempt in range(_MAX_RETRIES):
        temp = min(temperature + attempt * _TEMP_INCREMENT, _MAX_TEMP)

        # First attempt: full prompt (with reasoning), generous token budget.
        # Retries: concise prompt (no reasoning), reduced token budget to
        # cap damage from thinking loops.
        if attempt == 0:
            system_prompt = _DIAGNOSE_SYSTEM
            max_tokens = _DIAGNOSE_TOKENS
        else:
            print(f"[diagnose] retry {attempt}/{_MAX_RETRIES - 1} "
                  f"temp={temp:.2f}")
            system_prompt = _DIAGNOSE_SYSTEM_CONCISE
            max_tokens = _DIAGNOSE_CONCISE_TOKENS

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            raw, thinking = proxy.chat_completion(
                model, messages, temperature=temp, max_tokens=max_tokens
            )
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                continue
            raise

        parsed = _try_parse_output(raw)
        if parsed is not None:
            diagnoses, reasoning = parsed
            return diagnoses, reasoning, thinking

    print(f"[diagnose] all {_MAX_RETRIES} attempts failed; returning empty")
    return [], "", ""


def safe_search(diagnoses: list[str], k: int = 3) -> list[ICD10Match]:
    """Run :func:`search_icd10`, returning ``[]`` on error or invalid input.

    Filters out empty strings and ``[error]``-prefixed entries (emitted by
    node functions when the model call fails) so that error placeholders
    don't pollute the vector search results.
    """
    valid = [d for d in diagnoses if d and not d.startswith("[error]")]
    if not valid:
        return []
    try:
        return search_icd10(valid, k=k)
    except Exception:
        return []


@dataclass
class _Candidate:
    """A merged search candidate with vector and/or FTS signals.

    Used by :func:`_merge_and_score` to unify results from independent
    vector and FTS queries into a single ranked list.
    """

    code: str
    short_desc: str
    long_desc: str
    vec_sim: float = 0.0
    fts_rank: float = 0.0
    in_vector: bool = False
    in_fts: bool = False


def _fetch_vector(
    conn, query_vec_json: str, limit: int
) -> list[dict]:
    """Pure vector cosine-similarity search over billable ICD-10 codes.

    Args:
        conn: Active psycopg connection.
        query_vec_json: JSON-serialised embedding vector.
        limit: Maximum rows to return.

    Returns:
        Rows with keys ``code``, ``short_desc``, ``long_desc``, ``vec_sim``.
    """
    return conn.execute(
        """SELECT code, short_desc, long_desc,
                  1 - (embedding <=> %s::halfvec) AS vec_sim
           FROM icd10_codes
           WHERE is_billable AND embedding IS NOT NULL
           ORDER BY embedding <=> %s::halfvec
           LIMIT %s""",
        (query_vec_json, query_vec_json, limit),
    ).fetchall()  # type: ignore


def _fetch_fts_and(
    conn, query_text: str, query_vec_json: str, limit: int
) -> list[dict]:
    """Full-text search in **AND** mode — all query lexemes must appear.

    Vector similarity is computed alongside FTS rank for later re-scoring.

    Returns rows with keys ``code``, ``short_desc``, ``long_desc``,
    ``fts_rank``, ``vec_sim``.
    """
    return conn.execute(
        """SELECT code, short_desc, long_desc,
                  ts_rank_cd(search_tsv,
                             plainto_tsquery('english', %s)) AS fts_rank,
                  1 - (embedding <=> %s::halfvec) AS vec_sim
           FROM icd10_codes
           WHERE is_billable
             AND search_tsv @@ plainto_tsquery('english', %s)
             AND embedding IS NOT NULL
           ORDER BY fts_rank DESC
           LIMIT %s""",
        (query_text, query_vec_json, query_text, limit),
    ).fetchall()  # type: ignore


def _fetch_fts_or(
    conn, query_text: str, query_vec_json: str, limit: int
) -> list[dict]:
    """Full-text search in **OR** mode — any query lexeme may appear.

    Fallback for AND-mode stemming mismatches (e.g. "uterine" stems to
    ``uterin`` while the ICD-10 description uses ``uteri``).

    Returns rows with keys ``code``, ``short_desc``, ``long_desc``,
    ``fts_rank``, ``vec_sim``.
    """
    return conn.execute(
        """WITH q AS (
               SELECT to_tsquery('english',
                   regexp_replace(
                       regexp_replace(
                           plainto_tsquery('english', %s)::text,
                           '''', '', 'g'),
                       ' & ', ' | ', 'g')
               ) AS ts_or
           )
           SELECT c.code, c.short_desc, c.long_desc,
                  ts_rank_cd(c.search_tsv, q.ts_or) AS fts_rank,
                  1 - (c.embedding <=> %s::halfvec) AS vec_sim
           FROM icd10_codes c, q
           WHERE c.is_billable
             AND c.search_tsv @@ q.ts_or
             AND c.embedding IS NOT NULL
           ORDER BY fts_rank DESC
           LIMIT %s""",
        (query_text, query_vec_json, limit),
    ).fetchall()  # type: ignore


def _fetch_fts(
    conn, query_text: str, query_vec_json: str, limit: int
) -> tuple[list[dict], bool]:
    """Fetch FTS candidates, falling back from AND to OR mode.

    AND mode requires all lexemes to match — these are exact-term hits
    that bypass the :data:`_FTS_FLOOR` in :func:`_merge_and_score`.
    When AND fails (stemming mismatches), OR mode catches partial
    matches, but those are subject to the floor to filter noise.

    Returns ``(rows, is_and_mode)``.
    """
    rows = _fetch_fts_and(conn, query_text, query_vec_json, limit)
    if rows:
        return rows, True
    rows = _fetch_fts_or(conn, query_text, query_vec_json, limit)
    return rows, False


def _merge_and_score(
    vec_rows: list[dict], fts_rows: list[dict], k: int,
    fts_is_and: bool = True,
) -> list[ICD10Match]:
    """Merge vector + FTS candidates using **vector-primary, FTS-boost** scoring.

    Scoring rules:

    - **Vector + FTS agreement** (code in both result sets):
      ``vec_sim + _FTS_BOOST * norm_fts`` — the FTS match confirms the
      semantic match, pushing exact-term codes above near-misses.
    - **Vector only** (code not in FTS results): ``vec_sim`` unchanged.
    - **FTS only, AND mode** (exact term match, not in vector top-k):
      ``vec_sim + _FTS_BOOST * norm_fts``.  No floor — AND matches are
      high-confidence exact hits whose only issue is low embedding
      similarity (e.g. "unspecified" codes).
    - **FTS only, OR mode** (partial/stemming match): included only if
      ``vec_sim >= _FTS_FLOOR``, scored with ``_FTS_BOOST``.  Codes below
      the floor are excluded — this filters semantically irrelevant
      partial matches.

    Args:
        vec_rows: Candidates from :func:`_fetch_vector`.
        fts_rows: Candidates from :func:`_fetch_fts`.
        k: Maximum matches to return.
        fts_is_and: Whether FTS rows came from AND mode (exact match).

    Returns:
        Top-k :class:`ICD10Match` objects sorted by score descending.
    """
    cands: dict[str, _Candidate] = {}
    for r in vec_rows:
        cands[r["code"]] = _Candidate(  # type: ignore
            code=r["code"],
            short_desc=r["short_desc"],
            long_desc=r["long_desc"],
            vec_sim=float(r["vec_sim"]),
            in_vector=True,
        )
    for r in fts_rows:
        code = r["code"]  # type: ignore
        if code in cands:
            cands[code].fts_rank = float(r["fts_rank"])  # type: ignore
            cands[code].in_fts = True
        else:
            cands[code] = _Candidate(
                code=code,
                short_desc=r["short_desc"],  # type: ignore
                long_desc=r["long_desc"],  # type: ignore
                vec_sim=float(r["vec_sim"]),  # type: ignore
                fts_rank=float(r["fts_rank"]),  # type: ignore
                in_fts=True,
            )

    max_fts = max(
        (c.fts_rank for c in cands.values() if c.fts_rank > 0), default=0.0
    )

    scored: list[tuple[_Candidate, float]] = []
    for c in cands.values():
        norm_fts = c.fts_rank / max_fts if max_fts > 0 and c.fts_rank > 0 else 0.0

        if c.in_vector and c.in_fts:
            score = c.vec_sim + _FTS_BOOST * norm_fts
        elif c.in_vector:
            score = c.vec_sim
        elif fts_is_and or c.vec_sim >= _FTS_FLOOR:
            score = c.vec_sim + _FTS_BOOST * norm_fts
        else:
            continue

        scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        ICD10Match(
            code=c.code,
            short_desc=c.short_desc,
            long_desc=c.long_desc,
            similarity=round(score, 4),
        )
        for c, score in scored[:k]
    ]


def search_icd10(queries: list[str], k: int = 3) -> list[ICD10Match]:
    """Find billable ICD-10-CM codes matching multiple diagnosis queries.

    Uses **vector-primary, FTS-boost** hybrid search:

    1. **Vector search** — pgvector HNSW cosine-similarity over code
       embeddings.  Returns ``k * _CANDIDATE_MULT`` candidates.  This
       is the primary signal: semantic similarity ranks the most
       relevant codes first.
    2. **FTS search** — PostgreSQL full-text search (AND → OR fallback)
       over code descriptions.  Returns up to ``k * _CANDIDATE_MULT``
       candidates with ``ts_rank_cd`` and vector similarity computed
       in the same query.
    3. **Merge & score** — candidates appearing in both result sets get
       a score boost (``_FTS_BOOST * norm_fts``).  Vector-only candidates
       keep their raw ``vec_sim``.  FTS-only candidates are included only
       if their ``vec_sim`` exceeds ``_FTS_FLOOR`` (OR mode) or bypass
      the floor (AND mode — exact term matches).

    When ``_FTS_BOOST = 0`` (default), FTS candidates are fetched but
    do not affect ranking — results are identical to pure vector search.
    The FTS infrastructure is preserved for future use with Z-code
    filtering or tuned boost values.

    This preserves the strengths of vector search (semantic matching)
    while letting FTS boost exact-term matches that embedding models
    penalise (e.g. "unspecified" codes like C539).

    Results are deduplicated by code (keeping the highest score) across
    all queries and sorted by score descending.

    Args:
        queries: Diagnosis descriptions to match against ICD-10 codes.
        k: Maximum codes to return per query (default 3).

    Returns:
        Deduplicated matching codes ranked by score (highest first).
    """
    if not queries:
        return []

    vectors = proxy.embed(queries)
    cand_k = k * _CANDIDATE_MULT
    seen: dict[str, ICD10Match] = {}

    with get_pool().connection() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 200")
        for query_text, vec in zip(queries, vectors, strict=False):
            qv = json.dumps(vec)
            vec_rows = _fetch_vector(conn, qv, cand_k)
            fts_rows, fts_is_and = _fetch_fts(conn, query_text, qv, cand_k)
            for m in _merge_and_score(vec_rows, fts_rows, k, fts_is_and):
                if m.code not in seen or m.similarity > seen[m.code].similarity:
                    seen[m.code] = m

    return sorted(seen.values(), key=lambda m: m.similarity, reverse=True)
