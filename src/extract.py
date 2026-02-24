import json
from pathlib import Path
from typing import Dict, Any

RAW_PATH = Path("data/raw/fixtures_mock.json")

def extract_from_mock() -> Dict[str, Any]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Mock file not found: {RAW_PATH}")

    with RAW_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload

def count_extracted(payload: Dict[str, Any]) -> int:
    fixtures = payload.get("fixtures", [])
    # on considère 1 record = 1 match
    return len(fixtures)