## RAG system overview

This project uses a Retrieval-Augmented Generation (RAG) system to let you ask natural‑language questions about parcel / tabular data and get answers that are grounded in the actual rows of your dataset, not just the model’s general knowledge.

At a high level:

- The dataset (by default `us_parcel_dataset_2000.json` in `app/`, or a path from `DATASET_PATH`/`DATASET_JSON_PATH`) is loaded into an in‑memory index by `ExcelRAG` (`llm/rag.py`).
- Each row is converted into a compact text description (owner, parcel ID, acres, land value, address, etc.), then embedded using the local `sentence-transformers` model `all-MiniLM-L6-v2`.
- When the user asks a question, the system retrieves the most relevant rows and passes them, plus a schema summary, to an LLM (via `LLMClient`) to generate a final answer.

## What this RAG system is used for

The RAG system powers:

- **Question answering over the parcel dataset**  
  - Examples: “Show parcels above \$500,000 land value”, “Parcels between 0.5 and 2 acres”, “Vacant residential parcels owned by X”, “What is this dataset about?”.
- **ID / owner–based lookups**  
  - Direct parcel ID search and owner name based filtering.
- **Range and filter queries**  
  - Parses land value ranges, acreage ranges, land‑use categories, owner filters, and returns matching rows.
- **Data utilities**  
  - “Give me N rows”, “2k records”, owner lists, CSV‑style exports, and average land‑value calculations.
- **Small‑talk / behavioral chat**  
  - Detects purely conversational questions and routes them to a lightweight chat system prompt instead of the dataset.

Because answers are built from retrieved rows, the model can explain *why* something is true and include concrete examples, not just generic text.

## Main components

- **`ExcelRAG` (`llm/rag.py`)**
  - Loads data from JSON/CSV or user‑uploaded files (Excel, CSV, JSON, TXT, PDF, DOCX) and turns them into a list of records.
  - Builds an index: embeds each record into a vector and stores both the embedding and original metadata.
  - Implements:
    - `query`: hybrid retrieval (parcel ID → comparison filters → semantic similarity).
    - `query_by_comparison`: numeric and categorical filters for land value, land use, acres, owner name.
    - `answer`: orchestrates retrieval + LLM call, handles special intents (overview, “how many rows”, CSV, lists, averages, etc.).

- **Singleton accessor `get_rag()`**
  - Lazily creates one `ExcelRAG` instance over the default dataset and reuses it for all questions.

- **Per‑request RAG `get_rag_from_table(table)`**
  - Builds a one‑off `ExcelRAG` index from a user‑provided table (CSV/JSON string or list of dicts) so you can ask questions about an uploaded file only.

- **`ParcelDatasetAgent` (`llm/parcel_dataset_agent.py`)**
  - Thin agent wrapper used by the backend.
  - Method `run(question, top_k, table, conversation_history, user_name)` returns a `ParcelDatasetAgentResult` with:
    - `answer`: final, cleaned answer text.
    - `contexts`: the retrieved row contexts (metadata + snippet text) that grounded the answer.

## Typical flow

1. **Index creation**
   - At startup (first call), `get_rag()` loads the default dataset and calls `ExcelRAG.build_index()`.
2. **User asks a question**
   - Frontend sends a natural‑language question to the backend agent (`ParcelDatasetAgent.run`).
3. **Retrieval**
   - The agent uses `get_rag()` (or `get_rag_from_table` if a custom table was provided) to retrieve the most relevant rows for that question.
4. **LLM answer generation**
   - The selected rows, schema summary, and conversation history are packed into a prompt and sent to the LLM.
   - The LLM produces an answer that must stay consistent with the provided rows.
5. **Response to the UI**
   - Backend returns `answer` + `contexts` so the UI can show both the answer and the underlying supporting data.

## Why use RAG here?

- **Grounded, auditable answers**: every answer can be traced back to actual rows in your parcel / table data.
- **Better control over business logic**: custom parsing for land value, acreage, land use, owner names, record counts, exports, etc.
- **Works with your own files**: you can point the system at different datasets or uploaded tables without retraining any model.
- **Scales to many questions**: one shared index is reused for multiple queries, instead of re‑sending the whole file each time.

