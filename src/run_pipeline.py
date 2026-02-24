import uuid
from datetime import datetime
from sqlalchemy import text

from src.utils.db import get_engine
from src.utils.logger import get_logger

from src.extract import extract_from_mock, count_extracted
from src.transform import transform, count_loaded
from src.load import load_all

logger = get_logger("run_pipeline")


def main():
    run_id = uuid.uuid4()
    started_at = datetime.utcnow()
    engine = get_engine()

    try:
        logger.info("Connecting to database...")

        # 1) Test DB + insert STARTED
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

        # 2) Extract
        payload = extract_from_mock()
        extracted = count_extracted(payload)
        logger.info(f"Extracted matches: {extracted}")

        # 3) Transform
        transformed = transform(payload)
        planned_to_load = count_loaded(transformed)
        logger.info(f"Rows planned to load: {planned_to_load}")

        # 4) Load
        loaded = load_all(engine, transformed)
        logger.info(f"Loaded rows: {loaded}")

        # 5) Update run log -> SUCCESS
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

        # Log failure (update if STARTED exists, else insert)
        with engine.begin() as conn:
            updated = conn.execute(
                text("""
                    UPDATE pipeline_run_log
                    SET ended_at = :ended_at,
                        status = :status,
                        error_message = :error_message
                    WHERE run_id = :run_id
                """),
                {
                    "ended_at": ended_at,
                    "status": "FAILED",
                    "error_message": str(e),
                    "run_id": str(run_id),
                },
            ).rowcount

            if updated == 0:
                conn.execute(
                    text("""
                        INSERT INTO pipeline_run_log (
                            run_id, started_at, ended_at, status, error_message
                        )
                        VALUES (
                            :run_id, :started_at, :ended_at, :status, :error_message
                        )
                    """),
                    {
                        "run_id": str(run_id),
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "status": "FAILED",
                        "error_message": str(e),
                    },
                )

        raise


if __name__ == "__main__":
    main()