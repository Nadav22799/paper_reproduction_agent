"""SQLite-backed experiment tracking for the Paper Reproduction Agent.

Thread-safe: check_same_thread=False + a threading.Lock for all writes.
Zero external dependencies — uses Python's built-in sqlite3.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    paper_title       TEXT,
    status            TEXT NOT NULL DEFAULT 'queued',
    llm_provider      TEXT,
    critic_mode       TEXT,
    config_json       TEXT,
    metrics_json      TEXT,
    total_cost        REAL DEFAULT 0.0,
    total_tokens      INTEGER DEFAULT 0,
    duration_seconds  REAL,
    phases_completed  TEXT,
    accuracy          REAL,
    error_message     TEXT,
    storage_backend   TEXT DEFAULT 'local',
    cloned_repo_path  TEXT,
    archive_path      TEXT,
    started_at        TEXT,
    completed_at      TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_runs_paper   ON runs(paper_id);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


class ExperimentDB:
    """SQLite experiment tracking.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to ``experiments.db`` in the current working directory.
    """

    def __init__(self, db_path: str = "experiments.db") -> None:
        self._db_path = str(Path(db_path))
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def create_run(
        self,
        paper_id: str,
        config: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        critic_mode: Optional[str] = None,
    ) -> str:
        """Create a new run record and return its ID."""
        import os
        run_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs
                    (id, paper_id, status, llm_provider, critic_mode, config_json,
                     storage_backend, started_at, created_at)
                VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    paper_id,
                    llm_provider,
                    critic_mode,
                    json.dumps(config or {}),
                    os.getenv("STORAGE_BACKEND", "local"),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return run_id

    def update_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the status (and optionally error_message) of a run."""
        completed_at = datetime.utcnow().isoformat() if status in ("completed", "failed") else None
        with self._lock:
            self._conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    error_message = COALESCE(?, error_message),
                    completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (status, error_message, completed_at, run_id),
            )
            self._conn.commit()

    def update_metrics(
        self,
        run_id: str,
        metrics_json: str,
        total_cost: float,
        total_tokens: int,
        duration: float,
        accuracy: Optional[float],
        phases: List[str],
        paper_title: Optional[str] = None,
        cloned_repo_path: Optional[str] = None,
    ) -> None:
        """Update metrics after workflow completes."""
        with self._lock:
            self._conn.execute(
                """
                UPDATE runs
                SET metrics_json      = ?,
                    total_cost        = ?,
                    total_tokens      = ?,
                    duration_seconds  = ?,
                    accuracy          = ?,
                    phases_completed  = ?,
                    paper_title       = COALESCE(?, paper_title),
                    cloned_repo_path  = COALESCE(?, cloned_repo_path)
                WHERE id = ?
                """,
                (
                    metrics_json,
                    total_cost,
                    total_tokens,
                    duration,
                    accuracy,
                    json.dumps(phases),
                    paper_title,
                    cloned_repo_path,
                    run_id,
                ),
            )
            self._conn.commit()

    def archive_run(self, run_id: str, archive_path: str) -> None:
        """Record the archive location for a run's workspace."""
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET archive_path = ? WHERE id = ?",
                (archive_path, run_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a single run as a dict, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(
        self,
        paper_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return runs ordered by created_at DESC with optional filters."""
        clauses: List[str] = []
        params: List[Any] = []
        if paper_id:
            clauses.append("paper_id = ?")
            params.append(paper_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_runs_for_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Return all runs for a paper ordered by created_at DESC."""
        return self.list_runs(paper_id=paper_id, limit=100)

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all runs."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                                            AS total_runs,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status='running'   THEN 1 ELSE 0 END) AS running,
                AVG(total_cost)                                      AS avg_cost,
                SUM(total_cost)                                      AS total_cost,
                AVG(duration_seconds)                                AS avg_duration
            FROM runs
            """
        ).fetchone()
        stats = dict(row) if row else {}

        # Per-provider breakdown
        provider_rows = self._conn.execute(
            """
            SELECT llm_provider, COUNT(*) AS count, SUM(total_cost) AS cost
            FROM runs
            WHERE llm_provider IS NOT NULL
            GROUP BY llm_provider
            """
        ).fetchall()
        stats["provider_breakdown"] = [dict(r) for r in provider_rows]

        # Recent cost trend (last 10 completed runs)
        trend_rows = self._conn.execute(
            """
            SELECT id, paper_id, total_cost, created_at
            FROM runs
            WHERE status = 'completed'
            ORDER BY created_at DESC
            LIMIT 10
            """
        ).fetchall()
        stats["recent_cost_trend"] = [dict(r) for r in trend_rows]

        return stats
