import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

def _log_check(
    engine: Engine,
    run_id: str,
    name: str,
    status: str,
    metric_value: Optional[float] = None,
    threshold: Optional[float] = None,
    details: Optional[str] = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO data_quality_check
                (check_id, run_id, check_name, status, metric_value, threshold, details, created_at)
                VALUES (:check_id, :run_id, :check_name, :status, :metric_value, :threshold, :details, :created_at)
            """),
            {
                "check_id": str(uuid.uuid4()),
                "run_id": run_id,
                "check_name": name,
                "status": status,
                "metric_value": metric_value,
                "threshold": threshold,
                "details": details,
                "created_at": datetime.utcnow(),
            },
        )

def run_quality_checks(engine: Engine, run_id: str) -> None:
    """
    Checks simples mais très crédibles en contexte bancaire :
    - pas de doublons match_id
    - pas de stats joueur sans match
    - pass_accuracy dans [0,1]
    - minutes <= 120
    - volumes non nuls
    """
    # 1) fact_match unique
    with engine.begin() as conn:
        dup = conn.execute(text("""
            SELECT COUNT(*) AS dup_count
            FROM (
              SELECT match_id
              FROM fact_match
              GROUP BY match_id
              HAVING COUNT(*) > 1
            ) x
        """)).scalar_one()

    status = "PASS" if dup == 0 else "FAIL"
    _log_check(engine, run_id, "no_duplicate_match_id", status, float(dup), 0.0, f"duplicates={dup}")
    if status == "FAIL":
        raise ValueError("DQ FAIL: duplicate match_id detected")

    # 2) orphan player stats (match missing)
    with engine.begin() as conn:
        orphan = conn.execute(text("""
            SELECT COUNT(*)
            FROM fact_player_match_stats s
            LEFT JOIN fact_match m ON m.match_id = s.match_id
            WHERE m.match_id IS NULL
        """)).scalar_one()

    status = "PASS" if orphan == 0 else "FAIL"
    _log_check(engine, run_id, "no_orphan_player_stats", status, float(orphan), 0.0, f"orphans={orphan}")
    if status == "FAIL":
        raise ValueError("DQ FAIL: orphan player stats detected")

    # 3) pass_accuracy range
    with engine.begin() as conn:
        bad_pa = conn.execute(text("""
            SELECT COUNT(*)
            FROM fact_player_match_stats
            WHERE pass_accuracy IS NOT NULL
              AND (pass_accuracy < 0 OR pass_accuracy > 1)
        """)).scalar_one()

    status = "PASS" if bad_pa == 0 else "FAIL"
    _log_check(engine, run_id, "pass_accuracy_in_range", status, float(bad_pa), 0.0, f"bad_rows={bad_pa}")
    if status == "FAIL":
        raise ValueError("DQ FAIL: pass_accuracy out of range")

    # 4) minutes sanity
    with engine.begin() as conn:
        bad_min = conn.execute(text("""
            SELECT COUNT(*)
            FROM fact_player_match_stats
            WHERE minutes IS NOT NULL AND (minutes < 0 OR minutes > 120)
        """)).scalar_one()

    status = "PASS" if bad_min == 0 else "FAIL"
    _log_check(engine, run_id, "minutes_sanity", status, float(bad_min), 0.0, f"bad_rows={bad_min}")
    if status == "FAIL":
        raise ValueError("DQ FAIL: minutes out of range")

    # 5) volumes
    with engine.begin() as conn:
        match_cnt = conn.execute(text("SELECT COUNT(*) FROM fact_match")).scalar_one()
        stat_cnt = conn.execute(text("SELECT COUNT(*) FROM fact_player_match_stats")).scalar_one()

    _log_check(engine, run_id, "volume_fact_match_nonzero", "PASS" if match_cnt > 0 else "FAIL", float(match_cnt), 1.0)
    _log_check(engine, run_id, "volume_fact_player_stats_nonzero", "PASS" if stat_cnt > 0 else "FAIL", float(stat_cnt), 1.0)

    if match_cnt == 0 or stat_cnt == 0:
        raise ValueError("DQ FAIL: volumes are zero")