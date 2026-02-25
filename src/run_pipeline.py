import os
import uuid
from datetime import datetime
from sqlalchemy import text

from src.utils.db import get_engine
from src.utils.logger import get_logger

from src.extract import (
    extract_from_mock,
    extract_football_data_laliga_real_madrid,
    count_extracted,
)
from src.transform import transform, transform_football_data, count_loaded
from src.load import load_all
from src.quality import run_quality_checks

logger = get_logger("run_pipeline")


def main():
    run_id = uuid.uuid4()
    started_at = datetime.utcnow()
    engine = get_engine()

    extracted = 0
    loaded = 0

    try:
        logger.info("Connecting to database...")

        # Insert STARTED
        with engine.begin() as conn:
            conn.execute(text("SELECT 1;"))
            conn.execute(
                text("""
                    INSERT INTO pipeline_run_log (
                        run_id, started_at, status, extracted_count, loaded_count
                    )
                    VALUES (
                        :run_id, :started_at, :status, :extracted_count, :loaded_count
                    )
                """),
                {
                    "run_id": str(run_id),
                    "started_at": started_at,
                    "status": "STARTED",
                    "extracted_count": 0,
                    "loaded_count": 0,
                },
            )

        mode = os.getenv("PIPELINE_MODE", "api").lower()

        # Extract + Transform
        if mode == "mock":
            payload = extract_from_mock()
            extracted = count_extracted(payload)
            transformed = transform(payload)
        else:
            payload = extract_football_data_laliga_real_madrid()
            extracted = count_extracted(payload)  # counts matches
            transformed = transform_football_data(payload)

        planned = count_loaded(transformed)
        logger.info(f"Rows planned to load: {planned}")

        # Load
        loaded = load_all(engine, transformed)
        logger.info(f"Loaded rows: {loaded}")

        # Quality checks
        run_quality_checks(engine, str(run_id))
        logger.info("Data quality checks: PASS")

        # Update SUCCESS
        ended_at = datetime.utcnow()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE pipeline_run_log
                    SET ended_at = :ended_at,
                        status = :status,
                        extracted_count = :extracted_count,
                        loaded_count = :loaded_count,
                        error_message = NULL
                    WHERE run_id = :run_id
                """),
                {
                    "ended_at": ended_at,
                    "status": "SUCCESS",
                    "extracted_count": extracted,
                    "loaded_count": loaded,
                    "run_id": str(run_id),
                },
            )

        logger.info(f"Pipeline SUCCESS. run_id={run_id}")

    except Exception as e:
        logger.exception("Pipeline failed")
        ended_at = datetime.utcnow()

        with engine.begin() as conn:
            updated = conn.execute(
                text("""
                    UPDATE pipeline_run_log
                    SET ended_at = :ended_at,
                        status = :status,
                        extracted_count = :extracted_count,
                        loaded_count = :loaded_count,
                        error_message = :error_message
                    WHERE run_id = :run_id
                """),
                {
                    "ended_at": ended_at,
                    "status": "FAILED",
                    "extracted_count": extracted,
                    "loaded_count": loaded,
                    "error_message": str(e),
                    "run_id": str(run_id),
                },
            ).rowcount

            if updated == 0:
                conn.execute(
                    text("""
                        INSERT INTO pipeline_run_log (
                            run_id, started_at, ended_at, status, extracted_count, loaded_count, error_message
                        )
                        VALUES (
                            :run_id, :started_at, :ended_at, :status, :extracted_count, :loaded_count, :error_message
                        )
                    """),
                    {
                        "run_id": str(run_id),
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "status": "FAILED",
                        "extracted_count": extracted,
                        "loaded_count": loaded,
                        "error_message": str(e),
                    },
                )

        raise


if __name__ == "__main__":
    main()