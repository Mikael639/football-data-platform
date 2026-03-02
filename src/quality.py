from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class QualityCheckResult:
    name: str
    status: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: Optional[str] = None
    error_message: Optional[str] = None


def _log_check(engine: Engine, run_id: str, result: QualityCheckResult) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO data_quality_check
                (check_id, run_id, check_name, status, metric_value, threshold, details, created_at)
                VALUES (:check_id, :run_id, :check_name, :status, :metric_value, :threshold, :details, :created_at)
                """
            ),
            {
                "check_id": str(uuid.uuid4()),
                "run_id": run_id,
                "check_name": result.name,
                "status": result.status,
                "metric_value": result.metric_value,
                "threshold": result.threshold,
                "details": result.details,
                "created_at": datetime.utcnow(),
            },
        )


def _scalar(engine: Engine, sql: str) -> int:
    with engine.begin() as conn:
        return conn.execute(text(sql)).scalar_one()


def _run_duplicate_match_check(engine: Engine) -> QualityCheckResult:
    duplicate_count = _scalar(
        engine,
        """
        SELECT COUNT(*) AS dup_count
        FROM (
          SELECT match_id
          FROM fact_match
          GROUP BY match_id
          HAVING COUNT(*) > 1
        ) x
        """,
    )
    status = "PASS" if duplicate_count == 0 else "FAIL"
    return QualityCheckResult(
        name="no_duplicate_match_id",
        status=status,
        metric_value=float(duplicate_count),
        threshold=0.0,
        details=f"duplicates={duplicate_count}",
        error_message="DQ FAIL: duplicate match_id detected",
    )


def _run_orphan_player_stats_check(engine: Engine) -> QualityCheckResult:
    orphan_count = _scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM fact_player_match_stats s
        LEFT JOIN fact_match m ON m.match_id = s.match_id
        WHERE m.match_id IS NULL
        """,
    )
    status = "PASS" if orphan_count == 0 else "FAIL"
    return QualityCheckResult(
        name="no_orphan_player_stats",
        status=status,
        metric_value=float(orphan_count),
        threshold=0.0,
        details=f"orphans={orphan_count}",
        error_message="DQ FAIL: orphan player stats detected",
    )


def _run_pass_accuracy_check(engine: Engine) -> QualityCheckResult:
    invalid_rows = _scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM fact_player_match_stats
        WHERE pass_accuracy IS NOT NULL
          AND (pass_accuracy < 0 OR pass_accuracy > 1)
        """,
    )
    status = "PASS" if invalid_rows == 0 else "FAIL"
    return QualityCheckResult(
        name="pass_accuracy_in_range",
        status=status,
        metric_value=float(invalid_rows),
        threshold=0.0,
        details=f"bad_rows={invalid_rows}",
        error_message="DQ FAIL: pass_accuracy out of range",
    )


def _run_minutes_sanity_check(engine: Engine) -> QualityCheckResult:
    invalid_rows = _scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM fact_player_match_stats
        WHERE minutes IS NOT NULL AND (minutes < 0 OR minutes > 120)
        """,
    )
    status = "PASS" if invalid_rows == 0 else "FAIL"
    return QualityCheckResult(
        name="minutes_sanity",
        status=status,
        metric_value=float(invalid_rows),
        threshold=0.0,
        details=f"bad_rows={invalid_rows}",
        error_message="DQ FAIL: minutes out of range",
    )


def _run_volume_checks(engine: Engine, allow_empty_player_stats: bool) -> list[QualityCheckResult]:
    match_count = _scalar(engine, "SELECT COUNT(*) FROM fact_match")
    player_stats_count = _scalar(engine, "SELECT COUNT(*) FROM fact_player_match_stats")
    player_stats_ok = player_stats_count > 0 or allow_empty_player_stats
    player_stats_threshold = 0.0 if allow_empty_player_stats else 1.0

    return [
        QualityCheckResult(
            name="volume_fact_match_nonzero",
            status="PASS" if match_count > 0 else "FAIL",
            metric_value=float(match_count),
            threshold=1.0,
            details=f"rows={match_count}",
            error_message="DQ FAIL: volumes are zero",
        ),
        QualityCheckResult(
            name="volume_fact_player_stats_nonzero",
            status="PASS" if player_stats_ok else "FAIL",
            metric_value=float(player_stats_count),
            threshold=player_stats_threshold,
            details=(
                "empty allowed for football_data source"
                if allow_empty_player_stats and player_stats_count == 0
                else f"rows={player_stats_count}"
            ),
            error_message="DQ FAIL: volumes are zero",
        ),
    ]


def _default_checks(engine: Engine, allow_empty_player_stats: bool) -> list[QualityCheckResult]:
    checks = [
        _run_duplicate_match_check(engine),
        _run_orphan_player_stats_check(engine),
        _run_pass_accuracy_check(engine),
        _run_minutes_sanity_check(engine),
    ]
    checks.extend(_run_volume_checks(engine, allow_empty_player_stats))
    return checks


def _raise_on_failure(results: list[QualityCheckResult]) -> None:
    failed = next((result for result in results if result.status == "FAIL"), None)
    if failed and failed.error_message:
        raise ValueError(failed.error_message)


def run_quality_checks(engine: Engine, run_id: str, allow_empty_player_stats: bool = False) -> None:
    """
    Checks simples mais tres credibles en contexte bancaire :
    - pas de doublons match_id
    - pas de stats joueur sans match
    - pass_accuracy dans [0,1]
    - minutes <= 120
    - volumes non nuls
    """
    results = _default_checks(engine, allow_empty_player_stats)
    for result in results:
        _log_check(engine, run_id, result)
    _raise_on_failure(results)
