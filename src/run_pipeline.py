import uuid
from datetime import datetime
from sqlalchemy import text

from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")

def main():
    run_id = uuid.uuid4()
    started_at = datetime.utcnow()

    engine = get_engine()

    try:
        logger.info("Connecting to database...")
        with engine.begin() as conn:
            # test simple
            conn.execute(text("SELECT 1;"))

            # log start
            conn.execute(
                text("""
                    INSERT INTO pipeline_run_log (run_id, started_at, status, extracted_count, loaded_count)
                    VALUES (:run_id, :started_at, :status, :extracted_count, :loaded_count)
                """),
                {
                    "run_id": str(run_id),
                    "started_at": started_at,
                    "status": "STARTED",
                    "extracted_count": 0,
                    "loaded_count": 0,
                }
            )

        logger.info(f"DB OK. run_id={run_id}")

    except Exception as e:
        logger.exception("Pipeline failed")
        ended_at = datetime.utcnow()

        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO pipeline_run_log (run_id, started_at, ended_at, status, error_message)
                    VALUES (:run_id, :started_at, :ended_at, :status, :error_message)
                """),
                {
                    "run_id": str(run_id),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "status": "FAILED",
                    "error_message": str(e),
                }
            )

        raise

if __name__ == "__main__":
    main()