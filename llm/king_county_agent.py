from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from llm.client import LLMClient
from llm.prompts import KING_COUNTY_SYSTEM_PROMPT, build_king_county_user_prompt


@dataclass
class KingCountyToolResult:
    """
    Structured view outputs used by the agent.

    This is deliberately lightweight; you can extend these to represent
    real tools (e.g., database lookups, GIS queries, etc.).
    """

    name: str
    description: str
    notes: str = ""


@dataclass
class KingCountyValueAssessment:
    """
    Simple value assessment for a user's query/answer pair.

    This can be interpreted as:
    - label: qualitative assessment such as 'good_value', 'fair_value', 'poor_value'
    - score: numeric score between 0.0 and 1.0
    - explanation: short human-readable justification
    """

    label: str
    score: float
    explanation: str


@dataclass
class KingCountyAgentResult:
    answer: str
    reasoning: str | None = None
    tools_used: List[KingCountyToolResult] = field(default_factory=list)
    value_assessment: KingCountyValueAssessment | None = None


class KingCountyAgent:
    """
    Simple agentic wrapper around the LLM + a structured King County "view".

    Today this is a single-shot call into the LLM with some lightweight
    structured metadata about which "tools" (views) were relevant.

    You can evolve this into a true multi-step agent by:
    - Calling out to real tools (databases, APIs) based on the question
    - Maintaining conversation state / memory across turns
    - Implementing planning / reflection loops, etc.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant") -> None:
        self.model = model

    async def run(self, question: str, focus_areas: List[str]) -> KingCountyAgentResult:
        client = LLMClient()
        user_prompt = build_king_county_user_prompt(question=question, focus_areas=focus_areas)

        answer = await client.complete(
            model=self.model,
            system_prompt=KING_COUNTY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=120,
        )

        return KingCountyAgentResult(
            answer=answer,
            reasoning=None,
            tools_used=[],
            value_assessment=None,
        )

    @staticmethod
    def _infer_tools_used(focus_areas: List[str]) -> List[KingCountyToolResult]:
        """No domain views (property, zoning, etc.); keep responses short."""
        return []

