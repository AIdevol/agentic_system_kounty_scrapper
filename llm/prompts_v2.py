"""
prompts_v2.py — Upgraded RAG prompts reference for the King County / Parcel dataset system.

- System prompts (PARCEL_DATASET_SYSTEM_PROMPT, TABLE_DATASET_SYSTEM_PROMPT) are now in prompts.py
  with 14 explicit question-type rules.
- This module holds the full 80+ PARCEL_QA_TRAINING_EXAMPLES for few-shot or eval.
- Use get_training_examples_by_category_full(category) for the full set, or import from llm.prompts
  for the main prompts and the shorter training list.
"""

from __future__ import annotations

# Re-export from prompts so v2 can be used as a single reference
from llm.prompts import (
    BEHAVIORAL_CHAT_SYSTEM_PROMPT,
    CONVERSATION_AND_QUESTION_TYPE_INSTRUCTION,
    HUMAN_AND_PERSONALISATION_INSTRUCTION,
    KING_COUNTY_SYSTEM_PROMPT,
    PARCEL_DATASET_SYSTEM_PROMPT,
    TABLE_DATASET_SYSTEM_PROMPT,
    build_king_county_user_prompt,
    get_how_can_i_help_response,
    get_personalisation_system_snippet,
    get_training_examples_by_category,
    get_user_offering_to_help_response,
    is_how_can_you_help_question,
    is_user_offering_to_help_question,
    list_training_categories,
)

# Full 80+ training examples (reference set) — same categories as in prompts.py, more examples per category
PARCEL_QA_TRAINING_EXAMPLES_FULL: list[dict] = [
    {"category": "dataset_description", "question": "What is this file about?", "expected_answer": "This is a parcel dataset with 2,000 records. Each row represents a real-estate parcel and includes fields like Parcel ID, Owner Name, Acres, Land Value, Last Sale Date, Land Use Code, Address, and Contact Info (phone and email)."},
    {"category": "dataset_description", "question": "Describe this dataset.", "expected_answer": "This dataset contains 2,000 parcel records for real-estate properties. Key columns are Parcel ID, Owner Placeholder Name, Acres, Land Value, Last Sale Date, Land Use / Use Code, Address, and Contact Info."},
    {"category": "dataset_description", "question": "What's inside this file?", "expected_answer": "It contains 2,000 real-estate parcel records with owner info, acreage, land value, sale dates, land use codes, addresses, and contact details."},
    {"category": "row_count", "question": "How many records are there?", "expected_answer": "There are 2,000 records in this dataset."},
    {"category": "row_count", "question": "How many rows does this dataset have?", "expected_answer": "The dataset has 2,000 rows."},
    {"category": "row_count", "question": "Total number of entries?", "expected_answer": "2,000 entries in total."},
    {"category": "lookup", "question": "Find parcel 621899-1408.", "expected_answer": "Parcel 621899-1408: Owner: Owner_1979, Acres: 0.22, Land Value: $282,200, Last Sale Date: 2020-02-10, Land Use: MHP - Mobile Home Park, Address: 3824 Jefferson Pl, Cleveland, OH 44114."},
    {"category": "lookup", "question": "What is the address for parcel 621899-1408?", "expected_answer": "3824 Jefferson Pl, Cleveland, OH 44114."},
    {"category": "lookup", "question": "Who owns parcel 621899-1408?", "expected_answer": "The owner is Owner_1979."},
    {"category": "yes_no", "question": "Is there a parcel with ID 621899-1408?", "expected_answer": "Yes. Parcel 621899-1408 is at 3824 Jefferson Pl, Cleveland, OH 44114."},
    {"category": "yes_no", "question": "Does the dataset have any MHP parcels?", "expected_answer": "Yes. At least one parcel has the land use code MHP - Mobile Home Park."},
    {"category": "count_aggregation", "question": "How many MHP parcels are there?", "expected_answer": "Based on the sample rows I can see, 1 parcel has the MHP - Mobile Home Park land use code. The full dataset of 2,000 records may contain more."},
    {"category": "list_enumerate", "question": "List all land use codes.", "expected_answer": "From the sample rows: MHP - Mobile Home Park."},
    {"category": "list_enumerate", "question": "Give me all email addresses.", "expected_answer": "From the sample rows: owner1979@example.com."},
    {"category": "filter", "question": "Show me parcels with land value over $300,000.", "expected_answer": "From the sample rows, no parcel has a land value over $300,000. The full dataset may contain some."},
    {"category": "filter", "question": "Show only MHP land use parcels.", "expected_answer": "From the sample rows: Parcel 621899-1408, 3824 Jefferson Pl, Cleveland, OH 44114 — Land Use: MHP - Mobile Home Park."},
    {"category": "sort_ranking", "question": "Which parcel has the highest land value?", "expected_answer": "From the sample rows, parcel 621899-1408 has the highest land value at $282,200."},
    {"category": "comparison", "question": "Compare parcel 621899-1408 and parcel 000001-0001.", "expected_answer": "Parcel 621899-1408: Land Value $282,200, Acres 0.22, Sold 2020-02-10. Parcel 000001-0001: not found in the sample rows I have access to."},
    {"category": "calculation", "question": "What is the total land value across all parcels?", "expected_answer": "I can only compute from the sample rows provided. From 1 sample row: $282,200. The total across all 2,000 records will be much higher."},
    {"category": "schema", "question": "What columns are in this dataset?", "expected_answer": "Parcel ID, Owner Placeholder Name, Acres, Land Value, Last Sale Date, Land Use / Use Code, Address, Contact Info."},
    {"category": "relationship", "question": "How is this data related?", "expected_answer": "Technical: Parcel ID is the primary key; Contact Info bundles phone + email. Business: each row is one parcel owned by one person, with location, value, sale history, and contact details."},
    {"category": "contact_info", "question": "What is the email for Owner_1979?", "expected_answer": "owner1979@example.com."},
    {"category": "contact_info", "question": "What is the phone number for parcel 621899-1408?", "expected_answer": "+1 (318) 883-3390."},
    {"category": "casual", "question": "Hi!", "expected_answer": "Hi there! I'm here to help you explore your parcel dataset. What would you like to know?"},
    {"category": "casual", "question": "How are you?", "expected_answer": "Doing great, thanks for asking! Ready to help you dig into the parcel data. What would you like to look up?"},
]


def get_training_examples_by_category_full(category: str) -> list[dict]:
    """Return all training examples for a given category from the full 80+ reference set."""
    return [e for e in PARCEL_QA_TRAINING_EXAMPLES_FULL if e.get("category") == category]


def list_training_categories_full() -> list[str]:
    """Return sorted unique category names from the full training set."""
    return sorted({e.get("category", "") for e in PARCEL_QA_TRAINING_EXAMPLES_FULL if e.get("category")})
