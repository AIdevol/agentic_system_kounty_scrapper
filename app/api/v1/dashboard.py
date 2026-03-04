"""
Dashboard API: KPIs and chart data for property/parcel dataset (by location and price).
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter()

DEFAULT_DATASET_FILENAME = "us_parcel_dataset_2000.json"


def _get_dataset_path() -> str:
    path = os.getenv("DATASET_PATH") or os.getenv("DATASET_JSON_PATH")
    if not path:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        app_path = os.path.join(project_root, "app", DEFAULT_DATASET_FILENAME)
        root_path = os.path.join(project_root, DEFAULT_DATASET_FILENAME)
        path = app_path if os.path.exists(app_path) else root_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return path


def _parse_land_value(val: Any) -> float | None:
    if val is None:
        return None
    try:
        s = str(val).replace(",", "").strip()
        return float(s) if s else None
    except (ValueError, TypeError):
        return None


def _parse_location(address: Any) -> str:
    """Extract 'City, ST' from Address like '9035 Pine Ter, Queens, NY 11354'."""
    if not address:
        return "Unknown"
    s = str(address).strip()
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 2:
        # Last part is "ST zip"; second-to-last is city
        city = parts[-2].strip()
        st_zip = parts[-1].strip()
        # Take first 2 chars for state if "ST 12345" format
        st = st_zip.split()[0] if st_zip else ""
        return f"{city}, {st}" if st else city
    return s or "Unknown"


def _price_bucket(value: float) -> str:
    if value < 100_000:
        return "0–100k"
    if value < 500_000:
        return "100k–500k"
    if value < 1_000_000:
        return "500k–1M"
    return "1M+"


@router.get("/kpi", response_model=Dict[str, Any])
async def get_dashboard_kpi() -> Dict[str, Any]:
    """
    Return KPI and chart data for the parcel dataset:
    - by_location: list of { location, count, avg_land_value } sorted by avg_land_value desc
    - by_price_range: list of { range, count } for price buckets
    - total_parcels, avg_land_value_overall
    """
    path = _get_dataset_path()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="Dataset is not a list of records")

    by_location: Dict[str, List[float]] = defaultdict(list)
    by_price: Dict[str, int] = defaultdict(int)
    all_values: List[float] = []

    for row in data:
        if not isinstance(row, dict):
            continue
        addr = row.get("Address") or row.get("address")
        loc = _parse_location(addr)
        val = _parse_land_value(row.get("Land Value") or row.get("land_value"))
        if val is not None and val >= 0:
            by_location[loc].append(val)
            by_price[_price_bucket(val)] += 1
            all_values.append(val)

    # Build by_location summary (top 20 by avg value for chart)
    location_list: List[Dict[str, Any]] = [
        {
            "location": loc,
            "count": len(vals),
            "avg_land_value": round(sum(vals) / len(vals), 0),
        }
        for loc, vals in by_location.items()
    ]
    location_list.sort(key=lambda x: x["avg_land_value"], reverse=True)
    location_list = location_list[:20]

    price_order = ["0–100k", "100k–500k", "500k–1M", "1M+"]
    by_price_range: List[Dict[str, Any]] = [
        {"range": r, "count": by_price.get(r, 0)} for r in price_order
    ]

    return {
        "total_parcels": len(data),
        "avg_land_value_overall": round(sum(all_values) / len(all_values), 0) if all_values else 0,
        "by_location": location_list,
        "by_price_range": by_price_range,
    }
