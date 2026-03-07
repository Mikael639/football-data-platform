from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from statistics import median
from typing import Any

import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.logger import get_logger

logger = get_logger("alerts")


@dataclass(frozen=True)
class AlertEvaluation:
    enabled: bool
    severity: str
    reasons: list[str]
    summary: dict[str, Any]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _recent_success_loaded_counts(engine: Engine, *, exclude_run_id: str, limit: int = 8) -> list[float]:
    query = """
    SELECT loaded_count
    FROM pipeline_run_log
    WHERE status = 'SUCCESS'
      AND run_id::text <> :exclude_run_id
      AND loaded_count IS NOT NULL
      AND loaded_count > 0
    ORDER BY started_at DESC
    LIMIT :limit
    """
    with engine.begin() as conn:
        rows = conn.execute(text(query), {"exclude_run_id": str(exclude_run_id), "limit": int(limit)}).fetchall()
    return [float(row[0]) for row in rows if row and row[0] is not None]


def _latest_success_started_at(engine: Engine) -> datetime | None:
    query = """
    SELECT MAX(started_at)
    FROM pipeline_run_log
    WHERE status = 'SUCCESS'
    """
    with engine.begin() as conn:
        value = conn.execute(text(query)).scalar_one_or_none()
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def evaluate_pipeline_alert(
    *,
    engine: Engine,
    run_id: str,
    status: str,
    loaded_count: int | None,
    dq_results: list[Any] | None,
    freshness_days: int,
    error_message: str | None = None,
) -> AlertEvaluation:
    alerts_enabled = _env_bool("ALERTS_ENABLED", default=False)
    fail_count = 0
    warn_count = 0
    for result in dq_results or []:
        result_status = str(getattr(result, "status", "")).upper()
        if result_status == "FAIL":
            fail_count += 1
        elif result_status == "WARN":
            warn_count += 1

    reasons: list[str] = []
    severity = "INFO"
    now = datetime.now(timezone.utc)

    if str(status).upper() == "FAILED":
        severity = "CRITICAL"
        detail = f" ({error_message})" if error_message else ""
        reasons.append(f"Pipeline run failed{detail}.")

    if fail_count > 0:
        severity = "CRITICAL"
        reasons.append(f"Data quality has {fail_count} failing check(s).")
    elif warn_count > 0 and severity != "CRITICAL":
        severity = "WARNING"
        reasons.append(f"Data quality has {warn_count} warning check(s).")

    loaded_value = _safe_float(loaded_count)
    historical_loaded = _recent_success_loaded_counts(engine, exclude_run_id=run_id)
    if loaded_value is not None and loaded_value > 0 and len(historical_loaded) >= 3:
        baseline = median(historical_loaded)
        if baseline > 0 and loaded_value < 0.5 * baseline:
            if severity == "INFO":
                severity = "WARNING"
            reasons.append(
                f"Loaded rows anomaly: current={int(loaded_value)} below 50% of baseline={int(baseline)}."
            )

    latest_success = _latest_success_started_at(engine)
    if latest_success is None:
        if severity == "INFO":
            severity = "WARNING"
        reasons.append("No successful pipeline run found yet.")
    else:
        stale_hours = (now - latest_success).total_seconds() / 3600.0
        freshness_hours = max(int(freshness_days), 1) * 24
        if stale_hours > freshness_hours:
            if severity == "INFO":
                severity = "WARNING"
            reasons.append(
                f"Data freshness stale: last SUCCESS is {int(stale_hours)}h old (threshold {freshness_hours}h)."
            )

    summary = {
        "run_id": str(run_id),
        "status": str(status).upper(),
        "loaded_count": int(loaded_count or 0),
        "dq_fail_count": int(fail_count),
        "dq_warn_count": int(warn_count),
        "evaluated_at_utc": now.isoformat(timespec="seconds"),
    }
    return AlertEvaluation(enabled=alerts_enabled, severity=severity, reasons=reasons, summary=summary)


def _send_webhook(url: str, payload: dict[str, Any]) -> None:
    response = requests.post(url, json=payload, timeout=10)
    if response.status_code >= 400:
        raise RuntimeError(f"webhook status={response.status_code}")


def _send_smtp(
    *,
    host: str,
    port: int,
    username: str | None,
    password: str | None,
    sender: str,
    recipients: list[str],
    subject: str,
    body: str,
    starttls: bool,
) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    with smtplib.SMTP(host=host, port=port, timeout=10) as smtp:
        if starttls:
            smtp.starttls()
        if username:
            smtp.login(username, password or "")
        smtp.send_message(message)


def dispatch_pipeline_alert(
    *,
    engine: Engine,
    run_id: str,
    status: str,
    loaded_count: int | None,
    dq_results: list[Any] | None,
    freshness_days: int,
    error_message: str | None = None,
) -> None:
    evaluation = evaluate_pipeline_alert(
        engine=engine,
        run_id=run_id,
        status=status,
        loaded_count=loaded_count,
        dq_results=dq_results,
        freshness_days=freshness_days,
        error_message=error_message,
    )
    if not evaluation.enabled or not evaluation.reasons:
        return

    subject = f"[Football Data Platform] {evaluation.severity} run={evaluation.summary['run_id'][:8]}"
    reasons_lines = "\n".join(f"- {reason}" for reason in evaluation.reasons)
    body = (
        f"Severity: {evaluation.severity}\n"
        f"Status: {evaluation.summary['status']}\n"
        f"Run ID: {evaluation.summary['run_id']}\n"
        f"Loaded rows: {evaluation.summary['loaded_count']}\n"
        f"DQ FAIL: {evaluation.summary['dq_fail_count']} | DQ WARN: {evaluation.summary['dq_warn_count']}\n\n"
        f"Reasons:\n{reasons_lines}\n"
    )

    logger.warning("ALERT %s | %s", evaluation.severity, " | ".join(evaluation.reasons))

    webhook_url = (os.getenv("ALERT_WEBHOOK_URL") or "").strip()
    if webhook_url:
        payload = {
            "subject": subject,
            "severity": evaluation.severity,
            "summary": evaluation.summary,
            "reasons": evaluation.reasons,
            "message": body,
        }
        try:
            _send_webhook(webhook_url, payload)
        except Exception as exc:
            logger.error("Webhook alert failed: %s", exc)

    smtp_host = (os.getenv("ALERT_SMTP_HOST") or "").strip()
    smtp_to = (os.getenv("ALERT_SMTP_TO") or "").strip()
    smtp_from = (os.getenv("ALERT_SMTP_FROM") or "").strip()
    if smtp_host and smtp_to and smtp_from:
        recipients = [item.strip() for item in smtp_to.split(",") if item.strip()]
        if recipients:
            smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
            smtp_user = (os.getenv("ALERT_SMTP_USER") or "").strip() or None
            smtp_password = os.getenv("ALERT_SMTP_PASSWORD")
            smtp_starttls = _env_bool("ALERT_SMTP_STARTTLS", default=True)
            try:
                _send_smtp(
                    host=smtp_host,
                    port=smtp_port,
                    username=smtp_user,
                    password=smtp_password,
                    sender=smtp_from,
                    recipients=recipients,
                    subject=subject,
                    body=body,
                    starttls=smtp_starttls,
                )
            except Exception as exc:
                logger.error("SMTP alert failed: %s", exc)
