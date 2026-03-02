import json
from datetime import datetime

from src.config import Settings
from src.quality import QualityCheckResult
from src.run_pipeline import build_pipeline_run_payload


def test_pipeline_log_payload_contains_metrics_and_volumes():
    settings = Settings.from_env({"DATA_MODE": "mock"})
    transformed = {
        "dim_date": [{"date_id": "2024-08-18"}],
        "dim_team": [{"team_id": 1}],
        "dim_competition": [{"competition_id": 1}],
        "dim_player": [{"player_id": 10}],
        "fact_match": [{"match_id": 100}],
        "fact_player_match_stats": [{"match_id": 100, "player_id": 10}],
    }
    dq_results = [
        QualityCheckResult(
            name="data_freshness_recent_max_date",
            status="WARN",
            severity="WARN",
            details="historical dataset",
        )
    ]

    payload = build_pipeline_run_payload(
        settings=settings,
        status="SUCCESS",
        started_at=datetime(2026, 3, 2, 10, 0, 0),
        ended_at=datetime(2026, 3, 2, 10, 0, 5),
        extracted=3,
        loaded=6,
        timings_ms={"extract_ms": 100, "transform_ms": 200, "load_ms": 300, "dq_ms": 50},
        transformed=transformed,
        payload={"incremental_window": None},
        dq_results=dq_results,
        error_message=None,
    )

    metrics = json.loads(payload["metrics_jsonb"])
    volumes = json.loads(payload["volumes_jsonb"])

    assert payload["status"] == "SUCCESS"
    assert metrics["mode"] == "mock"
    assert metrics["extract_ms"] == 100
    assert metrics["dq_summary"] == {"PASS": 0, "WARN": 1, "FAIL": 0}
    assert volumes["rows_fact_match"] == 1
    assert volumes["rows_fact_player_match_stats"] == 1
    assert volumes["loaded_count"] == 6
