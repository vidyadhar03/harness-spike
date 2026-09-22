"""Background dispatch for concept generation jobs.

Only schedules work: the durable state machine, atomic ownership (a lease claimed in the store) and
recovery all live in harness.memory.concept_generation. Two schedulers racing for the same job is
harmless - only the one that wins the store's atomic claim does anything.
"""
from __future__ import annotations

import asyncio
import logging

from harness.memory.concept_generation import ConceptGenerationService

log = logging.getLogger(__name__)


class GenerationDispatcher:
    def __init__(self, service: ConceptGenerationService, *, max_concurrent: int, enabled: bool = True):
        self.service, self.enabled = service, enabled
        self._sema = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task] = set()
        self._sweeper: asyncio.Task | None = None

    def schedule(self, project_id: str, job_id: str) -> None:
        if not self.enabled:
            return
        task = asyncio.create_task(self._run(project_id, job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def start_sweeper(self, interval_s: float) -> None:
        """Periodically pick up jobs that are unowned or whose owner's lease has lapsed (a crashed
        worker, or a job another dispatcher never got to). Live leases are never touched - see
        ConceptGenerationService.recover. Two dispatchers sweeping the same store is safe."""
        if self.enabled and self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_forever(interval_s))

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None

    async def _sweep_forever(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                for pid, jid in await asyncio.to_thread(self.service.recover):
                    self.schedule(pid, jid)
            except Exception:
                log.exception("generation sweep failed; will retry")

    async def _run(self, project_id: str, job_id: str) -> None:
        async with self._sema:
            try:
                # Real thread, deliberately not cancelled on timeout - same reasoning as JobRunner:
                # the provider call/import must run to a recorded outcome, not be abandoned mid-write.
                await asyncio.to_thread(self.service.run, project_id, job_id)
            except Exception:
                log.exception("generation job %s crashed outside the worker's own handling", job_id)
