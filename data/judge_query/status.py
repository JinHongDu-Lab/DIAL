from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON status file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_status(path: Path) -> dict[str, Any] | None:
    """Read a status object, returning ``None`` for missing/partial files."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


#: States a run can no longer leave on its own. Anything else means a process
#: was still meant to be working when the status was last written.
TERMINAL_STATES = frozenset(
    {"completed", "completed_with_failures", "failed", "cancelled"}
)


def owning_machine_id() -> str | None:
    """Return the machine identifier to stamp on run status files.

    ``logs/runs/`` is shared by every machine collecting into a synced study
    directory, while ``pid`` is only meaningful on the machine that wrote it.
    Recording the writer's machine lets a reader tell "this run died here" from
    "this run belongs to another machine", instead of guessing from a pid that
    means nothing locally.

    Returns
    -------
    str or None
        Value of ``CAUSAL_JUDGE_MACHINE_ID``, or ``None`` when unset — matching
        :func:`causal_judge.experiment.configured_machine_id` without importing
        it, since ``experiment`` already imports this module.
    """

    value = os.environ.get("CAUSAL_JUDGE_MACHINE_ID")
    return value.strip() or None if value is not None else None


def result_state(result: dict[str, dict[str, int]]) -> str:
    """Return a truthful terminal state for aggregated judge counters."""

    failures = sum(
        int(values.get("failures", 0)) + int(values.get("input_failures", 0))
        for values in result.values()
    )
    successful = sum(
        int(values.get("successes", 0))
        + int(values.get("cached", 0))
        + int(values.get("skipped", 0))
        + int(values.get("validated", 0))
        for values in result.values()
    )
    if failures and successful:
        return "completed_with_failures"
    if failures:
        return "failed"
    return "completed"


class ProgressTracker:
    """Persist throttled experiment progress for CLI and web consumers."""

    def __init__(
        self,
        path: Path,
        *,
        study: str,
        run: str,
        judges: tuple[str, ...],
        conditions: list[dict[str, Any]],
        dry_run: bool,
        runnable_judges: tuple[str, ...] | None = None,
        skipped_judges: dict[str, list[str]] | None = None,
        kind: str = "study",
    ) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._last_write = 0.0
        self.state: dict[str, Any] = {
            "version": 1,
            "study": study,
            "run": run,
            "kind": kind,
            "state": "queued",
            "pid": os.getpid(),
            "machine_id": owning_machine_id(),
            "dry_run": dry_run,
            "judges": list(judges),
            "requested_judges": list(judges),
            "runnable_judges": list(runnable_judges or judges),
            "skipped_judges": skipped_judges or {},
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "finished_at": None,
            "current_condition": None,
            "conditions": [
                {
                    "index": index,
                    **condition,
                    "state": "queued",
                    "judges": {},
                }
                for index, condition in enumerate(conditions)
            ],
            "error": None,
        }
        self._write(force=True)

    def _write(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < 0.5:
            return
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.path, self.state)
        self._last_write = now

    def running(self) -> None:
        with self._lock:
            self.state["state"] = "running"
            self._write(force=True)

    def condition_started(self, index: int) -> None:
        with self._lock:
            self.state["current_condition"] = index
            self.state["conditions"][index]["state"] = "running"
            self._write(force=True)

    def judge_progress(
        self,
        condition_index: int,
        judge: str,
        values: dict[str, int],
        *,
        force: bool = False,
    ) -> None:
        with self._lock:
            existing = self.state["conditions"][condition_index]["judges"].setdefault(
                judge, {}
            )
            existing.update(values)
            self._write(force=force)

    def judge_batch_progress(
        self,
        condition_index: int,
        judge: str,
        values: dict[str, Any],
    ) -> None:
        """Persist provider batch lifecycle details for one judge."""

        with self._lock:
            existing = self.state["conditions"][condition_index]["judges"].setdefault(
                judge, {}
            )
            existing["batch"] = values
            self._write(force=True)

    def condition_completed(
        self,
        index: int,
        result: dict[str, dict[str, int]],
    ) -> None:
        with self._lock:
            condition = self.state["conditions"][index]
            condition["state"] = result_state(result)
            for judge, values in result.items():
                existing = condition["judges"].setdefault(judge, {})
                existing.update(values)
            self._write(force=True)

    def completed(self) -> None:
        with self._lock:
            combined: dict[str, dict[str, int]] = {}
            for condition in self.state["conditions"]:
                for judge, values in condition.get("judges", {}).items():
                    totals = combined.setdefault(judge, {})
                    for name, value in values.items():
                        if isinstance(value, int) and not isinstance(value, bool):
                            totals[name] = totals.get(name, 0) + value
            self.state["state"] = result_state(combined)
            self.state["current_condition"] = None
            self.state["finished_at"] = utc_now()
            self._write(force=True)

    def failed(self, error: BaseException) -> None:
        with self._lock:
            self.state["state"] = "failed"
            self.state["finished_at"] = utc_now()
            self.state["error"] = f"{type(error).__name__}: {error}"[:1000]
            self._write(force=True)


def mark_cancelled(path: Path) -> None:
    """Mark an existing run status as cancelled."""

    status = read_status(path) or {"version": 1}
    status.update(
        {
            "state": "cancelled",
            "updated_at": utc_now(),
            "finished_at": utc_now(),
            "error": None,
        }
    )
    atomic_write_json(path, status)
