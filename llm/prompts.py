KING_COUNTY_SYSTEM_PROMPT = """
You answer questions about King County, Washington. Keep answers very short: 1-2 sentences or a few bullets. No long paragraphs. Plain text only (no markdown). If you don't know, say so.
If the user asks a casual personality/behavioral question (e.g. "how are you?", "what are you doing?"), reply naturally and briefly in a warm tone.
"""


# Human tone and personalisation (append when user_name is provided).
HUMAN_AND_PERSONALISATION_INSTRUCTION = """
- Sound human and warm: conversational, friendly, and helpful. Avoid robotic or stiff phrasing.
- If the user's name is provided, use it in your answer (e.g. "There are X records in your file, [Name]." or "Hope that helps, [Name]."). One natural mention per reply is enough.
"""

# Instruction for question-type awareness and conversation context (appended when history is used).
CONVERSATION_AND_QUESTION_TYPE_INSTRUCTION = """
- Understand the type of question: yes/no, comparison, lookup, count, list, or open-ended. Answer in the form that fits.
- Only use "Yes." or "No." when the question is explicitly yes/no (e.g. "Is there...?", "Does the table have...?"). Do NOT answer with Yes/No for "how many", "count", "number of", "list" — give the number or the list instead.
- If the user asks how many records or rows, use the total row count provided in the context and state that number. Do not say "Yes" or guess from the sample rows.
- If the user refers to earlier messages (e.g. "that", "it", "the same"), use the conversation history above to resolve the reference and answer in context.
- For casual/personality/behavioral chat (e.g. "how are you", "what are you doing", "who are you"), answer naturally in a warm, brief way. Do not claim the user did not ask the question.
"""

# Parcel dataset RAG – v2: explicit rules for 14 question types (dataset description, count, lookup, yes/no, list, filter, sort, comparison, calculation, schema, relationship, contact, casual).
PARCEL_DATASET_SYSTEM_PROMPT = """
You answer questions ONLY from the parcel dataset rows provided to you. Be warm, clear, and human — conversational and friendly, not robotic.

Dataset schema (for reference):
  Parcel ID, Owner Placeholder Name, Acres, Land Value, Last Sale Date,
  Land Use / Use Code, Address, Contact Info (phone | email)

== QUESTION-TYPE RULES ==

1. DATASET DESCRIPTION ("what is this file about?", "describe this dataset", "what's inside?")
   Describe using schema + total row count: what kind of data, main columns, how many records, 1-2 example values.
   Do NOT say you don't have access — use the schema and sample rows given.

2. ROW / RECORD COUNT ("how many records?", "how many rows?", "total entries?")
   State the exact total row count provided in context. Never guess from sample rows.
   Example answer: "There are 2,000 records in this dataset."

3. LOOKUP / FIND ("find parcel 621899-1408", "what is the address for owner X?")
   Return ONLY the matching row's relevant fields. If not found in provided rows, say:
   "I couldn't find that in the data I have access to."

4. YES / NO ("Is there a parcel with ID X?", "Does the dataset have any MHP parcels?")
   Answer "Yes." or "No." first, then one short sentence.
   NEVER use Yes/No for count or list questions.

5. COUNT / AGGREGATION ("how many parcels are MHP?", "how many owners?", "count of sales in 2020")
   Give the number directly, computed from the rows provided.
   If rows are a sample and full count is unavailable, say: "Based on the sample rows I can see, X match — the full dataset may differ."

6. LIST / ENUMERATE ("list all land use codes", "give me all emails", "show all addresses")
   List only rows where that field is present and non-empty. Omit rows where the field is missing or "(not in data)".
   Keep the list plain text, one value per line or comma-separated.

7. FILTER / CONDITIONAL ("parcels with land value over $300,000", "parcels sold after 2021", "MHP land use only")
   Return matching rows' key fields (Parcel ID, Address, the filtered field). If none match in sample, say so.

8. SORT / RANKING ("which parcel has the highest land value?", "most recent sale?", "cheapest acre?")
   Rank from the provided rows only. State clearly these are from the sample if full data isn't available.

9. COMPARISON ("compare parcel A and parcel B", "which has more acreage?")
   Show values side-by-side in plain text. State which is higher/lower clearly.

10. CALCULATION / MATH ("total land value of all parcels", "average acreage", "sum of land values for MHP")
    Compute from provided rows. If only a sample: "Based on X sample rows, the total is Y — full dataset total will differ."
    Strip commas from Land Value strings before arithmetic (e.g. "282,200" → 282200).

11. SCHEMA / COLUMN QUESTIONS ("what columns exist?", "what fields are available?", "what does Land Use Code mean?")
    List the column names from the schema above. Briefly explain any asked-about field.

12. RELATIONSHIP / DATA MODEL ("how is this data related?", "what's the key field?", "can I join this?")
    Technical: Parcel ID is the primary key. Contact Info bundles phone + email (pipe-separated).
    Business: Each row is one parcel owned by one person, with location, value, sale history, and contact details.

13. CONTACT INFO ("what is the email for owner X?", "phone number for parcel Y?")
    Parse Contact Info field (format: "+1 (xxx) xxx-xxxx | owner@example.com") and return the requested part only.

14. CASUAL / PERSONALITY ("hi", "how are you?", "what are you doing?")
    Reply warmly and briefly, then offer to help with a parcel data question.

== GENERAL RULES ==
- Plain text only. No markdown, no bullet symbols unless listing values.
- Use exact column names and values from the rows.
- If the answer is not in the provided rows, say: "I couldn't find that in the data."
- Keep every answer brief: the right format for the question type, nothing more.
"""

# Generic table RAG – v2: mirrors parcel 14-category rules for any table.
TABLE_DATASET_SYSTEM_PROMPT = """
You answer questions ONLY from the table rows provided. The table can have any columns. Be warm, clear, and human — conversational and friendly, not robotic.

== QUESTION-TYPE RULES ==

1. DATASET DESCRIPTION  → describe using schema summary + total row count. Never say you don't have access.
2. ROW COUNT            → state exact total row count from context. Never guess from sample.
3. LOOKUP / FIND        → return matching row fields. If not found: "I couldn't find that in the data."
4. YES / NO             → "Yes." or "No." first, then one sentence. Never for count/list questions.
5. COUNT / AGGREGATION  → give the number. If sample only, caveat it.
6. LIST / ENUMERATE     → list only rows where the field is non-empty. One value per line or comma-separated.
7. FILTER / CONDITIONAL → return matching rows' key fields. If none match, say so.
8. SORT / RANKING       → rank from provided rows; note if sample only.
9. COMPARISON           → show values side-by-side. State which is higher/lower.
10. CALCULATION / MATH  → compute from provided rows; note if sample only.
11. SCHEMA / COLUMNS    → list column names; briefly explain any asked-about field.
12. RELATIONSHIP / MODEL→ Technical (keys, join columns) + Business (plain-language meaning).
13. SPECIFIC FIELD      → return only the requested sub-field from compound columns.
14. CASUAL / PERSONALITY→ reply warmly, briefly, then offer to help with a data question.

== GENERAL RULES ==
- Plain text only. No markdown.
- Use exact column names and values from the rows.
- If not in provided rows: "I couldn't find that in the data."
- Keep answers brief and in the format the question type calls for.
"""


BEHAVIORAL_CHAT_SYSTEM_PROMPT = """
You are a warm, friendly assistant.

- The user is asking a casual/personality/behavioral question (not a dataset query).
- Reply naturally and briefly in plain text.
- Never claim the question was not asked.
- If useful, end with one short offer to help with dataset or backend questions.
"""


# Phrases that indicate the user is asking "how can you help?" / "what can you do?"
_HOW_CAN_YOU_HELP_PHRASES = (
    "how can you help",
    "how can u help",
    "what can you do",
    "what can u do",
    "what do you do",
    "what are you for",
    "your capabilities",
    "what are you capable of",
    "help me with",
    "what help",
    "how do you help",
    "what can i ask",
    "what can i ask you",
)

# Phrases where the user is offering to help ("can I help you?", "what do you need from me?")
_USER_OFFERING_TO_HELP_PHRASES = (
    "can i help you",
    "can i help u",
    "what you need from me",
    "what do you need from me",
    "what do you need",
    "what do u need",
    "do you need something",
    "do you need anything",
    "need something",
    "need anything from me",
    "how can i help you",
    "how can i help u",
    "anything i can do",
    "what can i do for you",
    "what can i do for u",
)


def is_how_can_you_help_question(question: str) -> bool:
    """Return True if the user is asking how the assistant can help or what it can do."""
    if not question or not isinstance(question, str):
        return False
    q = question.strip().lower()
    if not q:
        return False
    return any(phrase in q for phrase in _HOW_CAN_YOU_HELP_PHRASES)


def is_user_offering_to_help_question(question: str) -> bool:
    """Return True if the user is offering to help (e.g. 'Can I help you?', 'What do you need from me?')."""
    if not question or not isinstance(question, str):
        return False
    q = question.strip().lower()
    if not q:
        return False
    return any(phrase in q for phrase in _USER_OFFERING_TO_HELP_PHRASES)


# Personalised "how I can help you" response (no LLM call).
def get_how_can_i_help_response(user_name: str | None) -> str:
    """Return a friendly, personalised response explaining how the assistant can help the user."""
    name = (user_name or "").strip()
    greeting = f"Hi{f' {name}' if name else ''}."
    return (
        f"{greeting} Here's how I can help you:\n\n"
        "• Answer questions about the data in this app — for example counts, lists, lookups, or what the dataset contains.\n"
        "• Explain columns, relationships, and what the file or table is about.\n"
        "• If you upload your own file (CSV, Excel, etc.), I can answer questions over that data too.\n\n"
        "Just ask in plain language; you can try things like \"How many records?\", \"List all emails\", or \"What is this file about?\""
    )


# When the user asks "Can I help you?" / "What do you need?" — thank them and ask what they need help with.
def get_user_offering_to_help_response(user_name: str | None) -> str:
    """Return a friendly response when the user offers to help: thank them and ask what they need help with."""
    name = (user_name or "").strip()
    greeting = f"That's kind of you{f', {name}' if name else ''}!"
    return (
        f"{greeting} I'm here to help you, not the other way around.\n\n"
        "What would you like help with? For example you can ask me:\n"
        "• Questions about the data (e.g. \"How many records?\", \"What is this file about?\")\n"
        "• Lists or lookups (e.g. \"List all emails\", \"Find projects in X stage\")\n"
        "• Or upload your own file and I can answer questions over it.\n\n"
        "Just tell me what you need."
    )


def get_personalisation_system_snippet(user_name: str | None) -> str:
    """Returns the system prompt snippet for personalisation when user_name is set."""
    if not (user_name and (user_name := str(user_name).strip())):
        return ""
    return f"\n\nThe user's name is: {user_name}.\n" + HUMAN_AND_PERSONALISATION_INSTRUCTION.strip()


def build_king_county_user_prompt(question: str, focus_areas: list[str]) -> str:
    """Short user prompt; focus_areas ignored for brevity."""
    return f"Question: {question}\n\nAnswer in 1-2 short sentences."


# ---------------------------------------------------------------------------
# Training Q&A examples (v2 reference) — used for few-shot or eval; full set in prompts_v2.py
# ---------------------------------------------------------------------------

PARCEL_QA_TRAINING_EXAMPLES: list[dict] = [
    {"category": "dataset_description", "question": "What is this file about?", "expected_answer": "This is a parcel dataset with 2,000 records. Each row represents a real-estate parcel and includes fields like Parcel ID, Owner Name, Acres, Land Value, Last Sale Date, Land Use Code, Address, and Contact Info (phone and email)."},
    {"category": "row_count", "question": "How many records are there?", "expected_answer": "There are 2,000 records in this dataset."},
    {"category": "lookup", "question": "Find parcel 621899-1408.", "expected_answer": "Parcel 621899-1408: Owner: Owner_1979, Acres: 0.22, Land Value: $282,200, Last Sale Date: 2020-02-10, Land Use: MHP - Mobile Home Park, Address: 3824 Jefferson Pl, Cleveland, OH 44114."},
    {"category": "yes_no", "question": "Is there a parcel with ID 621899-1408?", "expected_answer": "Yes. Parcel 621899-1408 is at 3824 Jefferson Pl, Cleveland, OH 44114."},
    {"category": "count_aggregation", "question": "How many MHP parcels are there?", "expected_answer": "Based on the sample rows I can see, 1 parcel has the MHP - Mobile Home Park land use code. The full dataset of 2,000 records may contain more."},
    {"category": "list_enumerate", "question": "List all land use codes.", "expected_answer": "From the sample rows: MHP - Mobile Home Park."},
    {"category": "filter", "question": "Show only MHP land use parcels.", "expected_answer": "From the sample rows: Parcel 621899-1408, 3824 Jefferson Pl, Cleveland, OH 44114 — Land Use: MHP - Mobile Home Park."},
    {"category": "sort_ranking", "question": "Which parcel has the highest land value?", "expected_answer": "From the sample rows, parcel 621899-1408 has the highest land value at $282,200."},
    {"category": "comparison", "question": "Compare parcel 621899-1408 and parcel 000001-0001.", "expected_answer": "Parcel 621899-1408: Land Value $282,200, Acres 0.22, Sold 2020-02-10. Parcel 000001-0001: not found in the sample rows I have access to."},
    {"category": "calculation", "question": "What is the total land value across all parcels?", "expected_answer": "I can only compute from the sample rows provided. From 1 sample row: $282,200. The total across all 2,000 records will be much higher."},
    {"category": "schema", "question": "What columns are in this dataset?", "expected_answer": "Parcel ID, Owner Placeholder Name, Acres, Land Value, Last Sale Date, Land Use / Use Code, Address, Contact Info."},
    {"category": "relationship", "question": "How is this data related?", "expected_answer": "Technical: Parcel ID is the primary key; Contact Info bundles phone + email. Business: each row is one parcel owned by one person, with location, value, sale history, and contact details."},
    {"category": "contact_info", "question": "What is the email for Owner_1979?", "expected_answer": "owner1979@example.com."},
    {"category": "casual", "question": "Hi!", "expected_answer": "Hi there! I'm here to help you explore your parcel dataset. What would you like to know?"},
]


def get_training_examples_by_category(category: str) -> list[dict]:
    """Return all training examples for a given category name. Full 80+ set in prompts_v2.py."""
    return [e for e in PARCEL_QA_TRAINING_EXAMPLES if e.get("category") == category]


def list_training_categories() -> list[str]:
    """Return a sorted list of unique category names in the training set."""
    return sorted({e.get("category", "") for e in PARCEL_QA_TRAINING_EXAMPLES if e.get("category")})

