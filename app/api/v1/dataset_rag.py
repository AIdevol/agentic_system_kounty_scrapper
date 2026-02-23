import json
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, List, Union

from llm.parcel_dataset_agent import ParcelDatasetAgent
from llm.rag import parse_uploaded_file


router = APIRouter()


class ConversationTurn(BaseModel):
    """Single turn in conversation history."""

    role: str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., description="Message content.")


class DatasetRAGQuery(BaseModel):
    """Request body for questions over the default dataset or a user-provided table."""

    question: str = Field(
        ...,
        description="Natural language question about the data (e.g. parcel dataset or your table).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many relevant rows to retrieve from the dataset.",
    )
    table: Union[str, List[Dict[str, Any]], List[List[Any]], None] = Field(
        default=None,
        description="Optional. Use this table instead of the default dataset. "
        "Can be: CSV string (first row = headers), JSON array of objects, or list of rows (first row = headers).",
    )
    conversation_history: List[ConversationTurn] | None = Field(
        default=None,
        description="Optional. Prior turns in this conversation so the agent understands question type and what was already asked.",
    )
    user_name: str | None = Field(
        default=None,
        description="Optional. User's name for a human, personalised reply (e.g. addressing them by name).",
    )


class DatasetContextRow(BaseModel):
    """Single retrieved row used to answer the question."""

    row_index: int
    metadata: Dict[str, Any]
    text: str


class DatasetRAGResponse(BaseModel):
    """LLM-grounded answer plus the underlying rows used."""

    answer: str
    contexts: List[DatasetContextRow]


@router.post("/query", response_model=DatasetRAGResponse)
async def query_parcel_dataset(request: Request) -> DatasetRAGResponse:
    """
    Ask a question over the default dataset or an uploaded file.
    - JSON body: { "question": "...", "top_k": 5, "table": null } for default dataset.
    - Multipart form: question, top_k, and optional file. When file is sent, it is used as the data source.
    """
    content_type = (request.headers.get("content-type") or "").strip().lower()

    # Treat as multipart when Content-Type is multipart (e.g. multipart/form-data; boundary=...)
    if "multipart" in content_type:
        # Require boundary so form parsing can succeed (browser sets this when using FormData)
        if "boundary=" not in content_type:
            raise HTTPException(
                status_code=400,
                detail="Multipart request must include boundary in Content-Type. When attaching a file, do not set Content-Type manually—let the browser set it.",
            )
        try:
            form = await request.form()
        except Exception as e:
            err_msg = str(e).strip() or repr(e)
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse multipart form: {err_msg}. Send multipart/form-data with 'question' and optional 'file'; do not set Content-Type manually when using FormData.",
            ) from e
        question = form.get("question")
        if question is None:
            raise HTTPException(status_code=400, detail="Missing 'question' in form.")
        if isinstance(question, bytes):
            question = question.decode("utf-8", errors="replace")
        elif not isinstance(question, str):
            v = getattr(question, "value", question)
            question = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        top_k_val = form.get("top_k", 5)
        if isinstance(top_k_val, bytes):
            top_k_val = top_k_val.decode("utf-8", errors="replace")
        elif hasattr(top_k_val, "value"):
            top_k_val = top_k_val.value
        try:
            top_k = int(top_k_val) if top_k_val is not None else 5
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(20, top_k))

        table = None
        file_part = form.get("file")
        if file_part is not None and hasattr(file_part, "read"):
            content = await file_part.read()
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is empty. Please choose a non-empty file (CSV, JSON, Excel, PDF, or DOCX).",
                )
            try:
                filename = getattr(file_part, "filename", None) or ""
                table = parse_uploaded_file(content, filename)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not parse file: {e}") from e
            if not table:
                raise HTTPException(
                    status_code=400,
                    detail="File has no data rows or could not be parsed. Use CSV (header row), JSON array of objects, Excel (.xlsx/.xls), PDF (text or tables), or Word (.docx).",
                )
        # Optional: conversation_history as JSON string in form
        conversation_history = None
        conv_raw = form.get("conversation_history")
        if conv_raw is not None:
            s = conv_raw
            if isinstance(s, bytes):
                s = s.decode("utf-8", errors="replace")
            elif hasattr(s, "value"):
                s = getattr(s, "value", s)
                if isinstance(s, bytes):
                    s = s.decode("utf-8", errors="replace")
            else:
                s = str(s)
            s = (s or "").strip()
            if s:
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        conversation_history = [
                            {"role": str(t.get("role", "user")), "content": str(t.get("content", ""))}
                            for t in arr
                            if isinstance(t, dict)
                        ]
                except json.JSONDecodeError:
                    pass
        # Optional: user name for personalisation
        user_name = None
        name_raw = form.get("user_name")
        if name_raw is not None:
            if isinstance(name_raw, bytes):
                user_name = name_raw.decode("utf-8", errors="replace").strip()
            elif hasattr(name_raw, "value"):
                v = name_raw.value
                user_name = (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)).strip()
            else:
                user_name = str(name_raw).strip()
            if not user_name:
                user_name = None
    else:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON body.") from e
        # Client may have sent FormData with Content-Type: application/json, so body can be the raw multipart string
        if isinstance(body, str) and body.strip().startswith("--"):
            raise HTTPException(
                status_code=400,
                detail="File uploads must be sent as multipart/form-data. Do not set Content-Type to application/json when attaching a file.",
            )
        try:
            payload = DatasetRAGQuery.model_validate(body)
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid request body. Send JSON with 'question' (string) and optional 'top_k' (1–20) and 'table'.",
            ) from e
        question = payload.question
        top_k = payload.top_k
        table = payload.table
        raw_history = payload.conversation_history
        conversation_history = (
            [{"role": t.role, "content": t.content} for t in raw_history]
            if raw_history
            else None
        )
        user_name = (payload.user_name or "").strip() or None

    agent = ParcelDatasetAgent(top_k=top_k)
    try:
        result = await agent.run(
            question=question,
            top_k=top_k,
            table=table,
            conversation_history=conversation_history,
            user_name=user_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DatasetRAGResponse(
        answer=result.answer,
        contexts=[
            DatasetContextRow(row_index=row["row_index"], metadata=row["metadata"], text=row["text"])
            for row in result.contexts
        ],
    )

