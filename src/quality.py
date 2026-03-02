from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class QualityContext:
    data_mode: str
    freshness_days: int
    today: date
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class QualityCheckResult:
    name: str
    status: str
    severity: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def _status_to_severity(status: str) -> str:
    return {"PASS": "INFO", "WARN": "WARN", "FAIL": "ERROR"}.get(status, "INFO")


def _result(
    *,
    name: str,
    status: str,
    metric_value: Optional[float] = None,
    threshold: Optional[float] = None,
    details: Optional[str] = None,
    error_message: Optional[str] = None,
) -> QualityCheckResult:
    return QualityCheckResult(
        name=name,
        status=status,
        severity=_status_to_severity(status),
        metric_value=metric_value,
        threshold=threshold,
        details=details,
        error_message=error_message,
    )


def _log_check(engine: Engine, run_id: str, result: QualityCheckResult) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO data_quality_check
                (
                    check_id,
                    run_id,
                    check_name,
                    status,
                    severity,
                    metric_value,
                    threshold,
                    details,
                    created_at
                )
                VALUES (
                    :check_id,
                    :run_id,
                    :check_name,
                    :status,
                    :severity,
                    :metric_value,
                    :threshold,
                    :details,
                    :created_at
                )
                """
            ),
            {
                "check_id": str(uuid.uuid4()),
                "run_id": run_id,
                "check_name": result.name,
                "status": result.status,
                "severity": result.severity,
                "metric_value": result.metric_value,
                "threshold": result.threshold,
                "details": result.details,
                "created_at": datetime.utcnow(),
            },
        )


def _scalar(engine: Engine, sql: str) -> int:
    with engine.begin() as conn:
        return conn.execute(text(sql)).scalar_one()


def _scalar_date(engine: Engine, sql: str) -> date | None:
    with engine.begin() as conn:
        return conn.execute(text(sql)).scalar_one_or_none()


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
    return _result(
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
    return _result(
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
    return _result(
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
    return _result(
        name="minutes_sanity",
        status=status,
        metric_value=float(invalid_rows),
        threshold=0.0,
        details=f"bad_rows={invalid_rows}",
        error_message="DQ FAIL: minutes out of range",
    )


def _run_team_dimension_check(engine: Engine, context: QualityContext) -> QualityCheckResult:
    team_count = _scalar(engine, "SELECT COUNT(*) FROM dim_team")
    if team_count > 0:
        status = "PASS"
    elif context.data_mode == "mock":
        status = "WARN"
    else:
        status = "FAIL"
    return _result(
        name="dim_team_nonempty",
        status=status,
        metric_value=float(team_count),
        threshold=1.0,
        details=f"rows={team_count}",
        error_message="DQ FAIL: dim_team is empty",
    )


def _run_match_volume_check(engine: Engine, context: QualityContext) -> QualityCheckResult:
    match_count = _scalar(engine, "SELECT COUNT(*) FROM fact_match")
    if match_count > 0:
        status = "PASS"
    elif context.data_mode == "mock":
        status = "WARN"
    else:
        status = "FAIL"
    return _result(
        name="fact_match_nonempty",
        status=status,
        metric_value=float(match_count),
        threshold=1.0,
        details=f"rows={match_count}",
        error_message="DQ FAIL: fact_match is empty",
    )


def _run_player_stats_volume_check(engine: Engine, allow_empty_player_stats: bool) -> QualityCheckResult:
    player_stats_count = _scalar(engine, "SELECT COUNT(*) FROM fact_player_match_stats")
    status = "PASS" if player_stats_count > 0 or allow_empty_player_stats else "FAIL"
    return _result(
        name="volume_fact_player_stats_nonzero",
        status=status,
        metric_value=float(player_stats_count),
        threshold=0.0 if allow_empty_player_stats else 1.0,
        details=(
            "empty allowed for football_data source"
            if allow_empty_player_stats and player_stats_count == 0
            else f"rows={player_stats_count}"
        ),
        error_message="DQ FAIL: fact_player_match_stats is empty",
    )


def _run_freshness_check(engine: Engine, context: QualityContext) -> QualityCheckResult:
    max_date = _scalar_date(engine, "SELECT MAX(date_id) FROM dim_date")
    if max_date is None:
        status = "WARN" if context.data_mode == "mock" else "FAIL"
        return _result(
            name="data_freshness_recent_max_date",
            status=status,
            details="max_date=NULL",
            error_message="DQ FAIL: no date available for freshness check",
        )

    threshold_date = context.today.fromordinal(context.today.toordinal() - context.freshness_days)
    if max_date >= threshold_date:
        status = "PASS"
    else:
        current_season_start = date(context.today.year if context.today.month >= 7 else context.today.year - 1, 7, 1)
        status = "WARN" if context.data_mode == "mock" or max_date < current_season_start else "FAIL"

    return _result(
        name="data_freshness_recent_max_date",
        status=status,
        metric_value=float(max_date.toordinal()),
        threshold=float(threshold_date.toordinal()),
        details=f"max_date={max_date.isoformat()} threshold={threshold_date.isoformat()}",
        error_message="DQ FAIL: max dim_date is stale",
    )


def _run_finished_score_consistency_check(context: QualityContext) -> QualityCheckResult:
    matches = (context.payload or {}).get("matches", [])
    if not matches:
        status = "WARN" if context.data_mode == "mock" else "PASS"
        return _result(
            name="finished_match_scores_consistent",
            status=status,
            details="no source matches available for status-based score check",
        )

    invalid_count = 0
    finished_count = 0
    for match in matches:
        if match.get("status") != "FINISHED":
            continue
        finished_count += 1
        score = (match.get("score") or {}).get("fullTime") or {}
        home_score = score.get("home")
        away_score = score.get("away")
        if home_score is None or away_score is None:
            invalid_count += 1
            continue
        if home_score < 0 or away_score < 0:
            invalid_count += 1

    if finished_count == 0:
        status = "WARN" if context.data_mode == "mock" else "PASS"
    else:
        status = "PASS" if invalid_count == 0 else "FAIL"

    return _result(
        name="finished_match_scores_consistent",
        status=status,
        metric_value=float(invalid_count),
        threshold=0.0,
        details=f"finished_matches={finished_count} invalid_rows={invalid_count}",
        error_message="DQ FAIL: finished matches have inconsistent scores",
    )


def _build_checks(
    engine: Engine,
    context: QualityContext,
    allow_empty_player_stats: bool,
) -> list[QualityCheckResult]:
    return [
        _run_duplicate_match_check(engine),
        _run_orphan_player_stats_check(engine),
        _run_pass_accuracy_check(engine),
        _run_minutes_sanity_check(engine),
        _run_team_dimension_check(engine, context),
        _run_match_volume_check(engine, context),
        _run_player_stats_volume_check(engine, allow_empty_player_stats),
        _run_freshness_check(engine, context),
        _run_finished_score_consistency_check(context),
    ]


def _raise_on_failure(results: list[QualityCheckResult]) -> None:
    failed = next((result for result in results if result.failed), None)
    if failed and failed.error_message:
        raise ValueError(failed.error_message)


def summarize_quality_results(results: list[QualityCheckResult]) -> dict[str, int]:
    summary = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def run_quality_checks(
    engine: Engine,
    run_id: str,
    allow_empty_player_stats: bool = False,
    context: QualityContext | None = None,
) -> list[QualityCheckResult]:
    quality_context = context or QualityContext(
        data_mode="api",
        freshness_days=7,
        today=datetime.utcnow().date(),
        payload=None,
    )
    results = _build_checks(engine, quality_context, allow_empty_player_stats)
    for result in results:
        _log_check(engine, run_id, result)
    _raise_on_failure(results)
    return results
