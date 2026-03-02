from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text

from src.config import Settings, get_settings
from src.extract import count_extracted, extract_csv, extract_football_data_laliga_all_clubs, extract_from_mock
from src.load import load_all
from src.quality import QualityCheckResult, QualityContext, run_quality_checks, summarize_quality_results
from src.transform import count_loaded, transform, transform_csv_to_tables, transform_football_data
from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _truncate_error(message: str | None, limit: int = 1000) -> str | None:
    if not message:
        return None
    if len(message) <= limit:
        return message
    return f"{message[: limit - 3]}..."


def build_volume_metrics(transformed: dict[str, list[dict[str, Any]]], *, extracted: int, loaded: int) -> dict[str, int]:
    return {
        "extracted_count": extracted,
        "loaded_count": loaded,
        "rows_dim_date": len(transformed.get("dim_date", [])),
        "rows_dim_team": len(transformed.get("dim_team", [])),
        "rows_dim_competition": len(transformed.get("dim_competition", [])),
        "rows_dim_player": len(transformed.get("dim_player", [])),
        "rows_fact_match": len(transformed.get("fact_match", [])),
        "rows_fact_player_match_stats": len(transformed.get("fact_player_match_stats", [])),
        "rows_fact_standings_snapshot": len(transformed.get("fact_standings_snapshot", [])),
    }


def build_runtime_metrics(
    settings: Settings,
    timings_ms: dict[str, int],
    payload: dict[str, Any] | None,
    dq_results: list[QualityCheckResult] | None,
) -> dict[str, Any]:
    total_duration = sum(timings_ms.values())
    return {
        "mode": settings.data_mode,
        "incremental": settings.incremental,
        "incremental_days": settings.incremental_days if settings.incremental else None,
        "incremental_window": (payload or {}).get("incremental_window"),
        "extract_ms": timings_ms.get("extract_ms", 0),
        "transform_ms": timings_ms.get("transform_ms", 0),
        "load_ms": timings_ms.get("load_ms", 0),
        "dq_ms": timings_ms.get("dq_ms", 0),
        "total_duration_ms": total_duration,
        "dq_summary": summarize_quality_results(dq_results or []),
    }


def build_pipeline_run_payload(
    *,
    settings: Settings,
    status: str,
    started_at: datetime,
    ended_at: datetime | None,
    extracted: int,
    loaded: int,
    timings_ms: dict[str, int],
    transformed: dict[str, list[dict[str, Any]]] | None,
    payload: dict[str, Any] | None,
    dq_results: list[QualityCheckResult] | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    transformed_rows = transformed or {}
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "extracted_count": extracted,
        "loaded_count": loaded,
        "error_message": _truncate_error(error_message),
        "metrics_jsonb": json.dumps(build_runtime_metrics(settings, timings_ms, payload, dq_results)),
        "volumes_jsonb": json.dumps(
            build_volume_metrics(
                transformed_rows,
                extracted=extracted,
                loaded=loaded,
            )
        ),
    }


def _insert_pipeline_run_start(run_id: str, payload: dict[str, Any], engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(text("SELECT 1;"))
        conn.execute(
            text(
                """
                INSERT INTO pipeline_run_log (
                    run_id,
                    started_at,
                    ended_at,
                    status,
                    extracted_count,
                    loaded_count,
                    error_message,
                    metrics_jsonb,
                    volumes_jsonb
                )
                VALUES (
                    :run_id,
                    :started_at,
                    :ended_at,
                    :status,
                    :extracted_count,
                    :loaded_count,
                    :error_message,
                    CAST(:metrics_jsonb AS JSONB),
                    CAST(:volumes_jsonb AS JSONB)
                )
                """
            ),
            {
                "run_id": run_id,
                **payload,
            },
        )


def _update_pipeline_run(run_id: str, payload: dict[str, Any], engine: Any) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text(
                """
                UPDATE pipeline_run_log
                SET ended_at = :ended_at,
                    status = :status,
                    extracted_count = :extracted_count,
                    loaded_count = :loaded_count,
                    error_message = :error_message,
                    metrics_jsonb = CAST(:metrics_jsonb AS JSONB),
                    volumes_jsonb = CAST(:volumes_jsonb AS JSONB)
                WHERE run_id = :run_id
                """
            ),
            {
                "run_id": run_id,
                **payload,
            },
        ).rowcount


def _run_extract(settings: Settings) -> tuple[dict[str, Any], int]:
    if settings.data_mode == "mock":
        payload = extract_from_mock()
        return payload, count_extracted(payload)
    if settings.data_mode == "csv":
        payload = extract_csv(settings=settings)
        return payload, count_extracted(payload)

    payload = extract_football_data_laliga_all_clubs(settings=settings)
    return payload, count_extracted(payload)


def _run_transform(settings: Settings, payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if settings.data_mode == "mock":
        return transform(payload)
    if settings.data_mode == "csv":
        return transform_csv_to_tables(payload)
    return transform_football_data(payload)


def _log_incremental_mode(settings: Settings, payload: dict[str, Any] | None) -> None:
    if settings.incremental:
        logger.info("Incremental mode ON. window=%s", (payload or {}).get("incremental_window"))
    else:
        logger.info("Incremental mode OFF. Full current-season extraction enabled.")


def main() -> None:
    settings = get_settings()
    settings.validate_for_pipeline()

    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    engine = get_engine(settings=settings)

    extracted = 0
    loaded = 0
    payload: dict[str, Any] | None = None
    transformed: dict[str, list[dict[str, Any]]] | None = None
    dq_results: list[QualityCheckResult] | None = None
    timings_ms = {
        "extract_ms": 0,
        "transform_ms": 0,
        "load_ms": 0,
        "dq_ms": 0,
    }

    start_payload = build_pipeline_run_payload(
        settings=settings,
        status="STARTED",
        started_at=started_at,
        ended_at=None,
        extracted=0,
        loaded=0,
        timings_ms=timings_ms,
        transformed={},
        payload=None,
        dq_results=[],
        error_message=None,
    )

    try:
        logger.info("Connecting to database...")
        _insert_pipeline_run_start(run_id, start_payload, engine)

        extract_started = time.perf_counter()
        payload, extracted = _run_extract(settings)
        timings_ms["extract_ms"] = _elapsed_ms(extract_started)
        _log_incremental_mode(settings, payload)

        transform_started = time.perf_counter()
        transformed = _run_transform(settings, payload)
        timings_ms["transform_ms"] = _elapsed_ms(transform_started)

        planned = count_loaded(transformed)
        logger.info("Rows planned to load: %s", planned)

        load_started = time.perf_counter()
        loaded = load_all(engine, transformed)
        timings_ms["load_ms"] = _elapsed_ms(load_started)
        logger.info("Loaded rows: %s", loaded)

        dq_started = time.perf_counter()
        dq_results = run_quality_checks(
            engine,
            run_id,
            allow_empty_player_stats=(settings.data_mode != "mock"),
            context=QualityContext(
                data_mode=settings.data_mode,
                freshness_days=settings.dq_freshness_days,
                today=datetime.utcnow().date(),
                payload=payload,
            ),
        )
        timings_ms["dq_ms"] = _elapsed_ms(dq_started)
        logger.info("Data quality checks summary: %s", summarize_quality_results(dq_results))

        end_payload = build_pipeline_run_payload(
            settings=settings,
            status="SUCCESS",
            started_at=started_at,
            ended_at=datetime.utcnow(),
            extracted=extracted,
            loaded=loaded,
            timings_ms=timings_ms,
            transformed=transformed,
            payload=payload,
            dq_results=dq_results,
            error_message=None,
        )
        _update_pipeline_run(run_id, end_payload, engine)
        logger.info("Pipeline SUCCESS. run_id=%s", run_id)

    except Exception as exc:
        logger.exception("Pipeline failed")
        failed_payload = build_pipeline_run_payload(
            settings=settings,
            status="FAILED",
            started_at=started_at,
            ended_at=datetime.utcnow(),
            extracted=extracted,
            loaded=loaded,
            timings_ms=timings_ms,
            transformed=transformed,
            payload=payload,
            dq_results=dq_results,
            error_message=str(exc),
        )
        updated = _update_pipeline_run(run_id, failed_payload, engine)
        if updated == 0:
            _insert_pipeline_run_start(run_id, failed_payload, engine)
        raise


if __name__ == "__main__":
    main()
