"""
Dataset agent: answers questions over the Projects CSV dataset using RAG.

Uses app/us_parcel_dataset_2000.json by default.
Organized like the King County agent: a single entry point that uses the RAG index
and returns a structured result (answer + contexts) for the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from llm.rag import get_rag, get_rag_from_table


@dataclass
class ParcelDatasetAgentResult:
    """Structured result from the parcel dataset agent."""

    answer: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)


class ParcelDatasetAgent:
    """
    Agent that answers questions using the indexed Projects dataset (CSV RAG).

    Uses the same pattern as KingCountyAgent: run(question, top_k) returns
    a structured result with answer text and the retrieved row contexts.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.default_top_k = top_k

    async def run(
        self,
        question: str,
        top_k: int | None = None,
        table: Union[str, List[Any], None] = None,
        conversation_history: List[Dict[str, str]] | None = None,
        user_name: str | None = None,
    ) -> ParcelDatasetAgentResult:
        """
        Answer a question using the default dataset or a user-provided table.

        Args:
            question: Natural language question about the data.
            top_k: Number of relevant rows to retrieve (default from __init__).
            table: Optional. If provided, the question is answered over this table only.
                   Can be: CSV string (first row = headers), JSON array of objects,
                   or list of dicts / list of rows. If omitted, uses default parcel dataset.
            conversation_history: Optional. Prior turns [{role, content}, ...] so the agent
                   understands question type and what was already asked in this conversation.
            user_name: Optional. User's name for a human, personalised tone (e.g. addressing them by name).

        Returns:
            ParcelDatasetAgentResult with answer and context rows.
        """
        k = top_k if top_k is not None else self.default_top_k
        if table is not None:
            rag = get_rag_from_table(table)
        else:
            rag = get_rag()
        result = await rag.answer(
            question=question,
            top_k=k,
            conversation_history=conversation_history,
            user_name=user_name,
        )

        # Strip markdown bold for consistent plain-text display
        answer_text = (result.get("answer") or "").replace("**", "")

        return ParcelDatasetAgentResult(
            answer=answer_text,
            contexts=result.get("contexts") or [],
        )
