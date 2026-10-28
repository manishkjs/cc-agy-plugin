"""Dependency-ordered, bounded-parallel execution of file tasks.

Rules: a task starts only after every task it depends on finished as ``landed`` or ``unchanged``;
two tasks on the same path never run at once; a task whose dependency failed (or was held) is
``held`` without running; an exception inside ``run`` fails that task, not the batch.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
from typing import Callable

from server.batch_plan import FileTask

DONE_STATES = frozenset({"landed", "unchanged"})


def run_task_graph(
    tasks: list[FileTask] | tuple[FileTask, ...],
    run: Callable[[FileTask], str],
    *,
    max_parallel: int = 4,
    on_settle: Callable[[FileTask, str], None] | None = None,
    warm_first: bool = False,
    stop: threading.Event | None = None,
) -> dict[str, str]:
    """Run tasks honoring depends_on and return {task_id: landed|unchanged|failed|held}."""
    by_id = {t.task_id: t for t in tasks}
    order = [t.task_id for t in tasks]
    states: dict[str, str] = {}
    running: dict[Future[str], FileTask] = {}
    busy_paths: set[str] = set()
    width = max(1, int(max_parallel))
    warming = bool(warm_first) and len(order) > 1

    def settle(task: FileTask, state: str) -> None:
        states[task.task_id] = state
        if on_settle is not None:
            on_settle(task, state)

    def safe_run(task: FileTask) -> str:
        try:
            result = run(task)
        except Exception:  # noqa: BLE001 - one task's crash must not take down the batch
            return "failed"
        # "held" means the batch stopped (budget, cancel) before the task could finish: it is not a
        # failure and a resumed run picks it up again.
        return result if result in ("landed", "unchanged", "failed", "held") else "failed"

    with ThreadPoolExecutor(max_workers=width, thread_name_prefix="cc-agy-batch") as pool:
        while len(states) < len(order):
            progressed = False
            for tid in order:
                if tid in states or any(f_task.task_id == tid for f_task in running.values()):
                    continue
                task = by_id[tid]
                deps = [d for d in task.depends_on if d in by_id]
                if any(states.get(d) in ("failed", "held") for d in deps) or (stop is not None and stop.is_set()):
                    settle(task, "held")
                    progressed = True
                    continue
                if not all(states.get(d) in DONE_STATES for d in deps):
                    continue
                limit = 1 if warming else width
                if len(running) >= limit or task.path in busy_paths:
                    continue
                busy_paths.add(task.path)
                running[pool.submit(safe_run, task)] = task
                progressed = True
            if not running:
                if not progressed:
                    # Nothing can start and nothing is running: remaining tasks wait on missing ids.
                    for tid in order:
                        if tid not in states:
                            settle(by_id[tid], "held")
                continue
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                task = running.pop(fut)
                busy_paths.discard(task.path)
                settle(task, fut.result())
            warming = False
    return states
