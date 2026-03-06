from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PIPELINE_PID_FILE = Path(os.getenv("FDP_PIPELINE_PID_FILE", "/tmp/fdp_pipeline.pid"))
PIPELINE_LOG_FILE = Path(os.getenv("FDP_PIPELINE_LOG_FILE", "/tmp/fdp_pipeline.log"))
PIPELINE_WORKDIR = Path(os.getenv("FDP_PIPELINE_WORKDIR", "/app"))


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def get_pipeline_process_status() -> dict[str, Any]:
    pid: int | None = None
    if PIPELINE_PID_FILE.exists():
        try:
            pid = int(PIPELINE_PID_FILE.read_text(encoding="utf-8").strip())
        except (TypeError, ValueError):
            pid = None

    running = bool(pid and _pid_is_running(int(pid)))
    if not running and PIPELINE_PID_FILE.exists():
        PIPELINE_PID_FILE.unlink(missing_ok=True)
        pid = None

    return {
        "running": running,
        "pid": int(pid) if pid is not None else None,
        "log_file": str(PIPELINE_LOG_FILE),
    }


def start_pipeline_process() -> dict[str, Any]:
    status = get_pipeline_process_status()
    if status["running"]:
        return {"started": False, "reason": "already_running", "pid": status.get("pid")}

    PIPELINE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== Pipeline triggered from dashboard at {datetime.utcnow().isoformat()}Z ===\n")

    stdout_handle = PIPELINE_LOG_FILE.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.run_pipeline"],
            cwd=str(PIPELINE_WORKDIR) if PIPELINE_WORKDIR.exists() else None,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    finally:
        stdout_handle.close()

    PIPELINE_PID_FILE.write_text(str(int(proc.pid)), encoding="utf-8")
    return {"started": True, "pid": int(proc.pid)}


def read_pipeline_log_tail(max_lines: int = 80) -> str:
    if not PIPELINE_LOG_FILE.exists():
        return "No pipeline log file yet."
    try:
        content = PIPELINE_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "Unable to read pipeline log file."
    return "\n".join(content[-max(int(max_lines), 1) :])
