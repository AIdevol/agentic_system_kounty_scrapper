from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from llm.king_county_agent import (
    KingCountyAgent,
    KingCountyToolResult,
    KingCountyValueAssessment,
)


router = APIRouter()


class KingCountyQuery(BaseModel):
    """Request body for querying the King County agent."""

    question: str = Field(..., description="User question about King County.")
    focus_areas: Optional[List[str]] = Field(default=None, description="Optional; currently unused. Kept for API compatibility.")
    streaming: bool = Field(
        default=False,
        description="Reserved for future use; if true, enable streaming responses.",
    )


class KingCountyValueAssessmentModel(BaseModel):
    """Structured view of how much 'value' the answer provides to the user."""

    label: str = Field(
        ...,
        description="Qualitative label such as 'good_value', 'fair_value', or 'poor_value'.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numeric score between 0.0 (no value) and 1.0 (excellent value).",
    )
    explanation: str = Field(
        ...,
        description="Short human-readable explanation of the assessment.",
    )


class KingCountyResponse(BaseModel):
    """High-level answer from the agent with optional structured tool outputs."""

    answer: str
    reasoning: Optional[str] = None
    tools_used: List[KingCountyToolResult] = Field(default_factory=list)
    value_assessment: Optional[KingCountyValueAssessmentModel] = Field(
        default=None,
        description="Optional assessment of how much value the answer provides.",
    )


@router.post("/query", response_model=KingCountyResponse)
async def query_king_county_agent(payload: KingCountyQuery) -> KingCountyResponse:
    """Single short answer from the King County agent. No property/zoning/views."""
    agent = KingCountyAgent()
    try:
        result = await agent.run(
            question=payload.question,
            focus_areas=payload.focus_areas or [],
        )
    except RuntimeError as exc:
        # Likely configuration issue such as missing API key
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    value_assessment: Optional[KingCountyValueAssessmentModel] = None
    if isinstance(result.value_assessment, KingCountyValueAssessment):
        va = result.value_assessment
        value_assessment = KingCountyValueAssessmentModel(
            label=va.label,
            score=va.score,
            explanation=(va.explanation or "").replace("**", ""),
        )

    # Strip markdown bold (**) so the UI shows plain text without asterisks
    answer_text = (result.answer or "").replace("**", "")

    return KingCountyResponse(
        answer=answer_text,
        reasoning=result.reasoning,
        tools_used=result.tools_used,
        value_assessment=value_assessment,
    )

