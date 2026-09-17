from __future__ import annotations

import os
from dataclasses import dataclass

from harness.memory.ingest import LOCK_RENEW_INTERVAL_S, LOCK_STALE_AFTER_S


@dataclass(frozen=True)
class ApiSettings:
    """API-specific config, separate from harness.memory.config.Settings (GCP/model config).

    Defaults are deliberately localhost-only: this server has no user-auth model (see
    API_CONTRACT.md), so the network boundary (bind address, allowed hosts, allowed
    CORS origins) is the only real access control in Phase 1.
    """
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    max_concurrent_jobs: int = 2
    # Bounds the one-time startup Firestore check (job/lock recovery - see main.lifespan).
    # Not a general per-request RPC timeout. 15s is generous for a normal Firestore call
    # (typically well under 1s) and drastically shorter than what a broken ADC credential
    # can silently retry for internally (observed: ~300s) before raising on its own -
    # see main.py's lifespan for why that gap matters and how it's handled.
    startup_timeout_s: float = 15.0
    # Enforced while receiving a POST /projects/{id}/sources upload, not after the fact -
    # see routes/sources.py. Nothing in the domain enforces an upload size limit at all
    # (Settings.inline_limit_bytes governs LLM inlining, not upload acceptance), so this
    # is a boundary-only addition.
    max_upload_bytes: int = 50 * 1024 * 1024
    # Ingest lock timing (harness.memory.ingest.LOCK_STALE_AFTER_S/LOCK_RENEW_INTERVAL_S
    # by default - same values the CLI uses, so the API and the CLI treat the shared
    # lock identically). Overridable mainly for tests that simulate a short expiry -
    # see tests/test_api.py's lock-renewal regression tests.
    ingest_lock_stale_after_s: float = LOCK_STALE_AFTER_S
    ingest_lock_renew_interval_s: float = LOCK_RENEW_INTERVAL_S

    @classmethod
    def from_env(cls) -> "ApiSettings":
        env = os.environ
        origins = env.get("HARNESS_API_ALLOWED_ORIGINS")
        hosts = env.get("HARNESS_API_ALLOWED_HOSTS")
        return cls(
            host=env.get("HARNESS_API_HOST", cls.host),
            port=int(env.get("HARNESS_API_PORT", cls.port)),
            allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()) if origins
                           else cls.allowed_origins,
            allowed_hosts=tuple(h.strip() for h in hosts.split(",") if h.strip()) if hosts
                         else cls.allowed_hosts,
            max_concurrent_jobs=int(env.get("HARNESS_API_MAX_CONCURRENT_JOBS", cls.max_concurrent_jobs)),
            startup_timeout_s=float(env.get("HARNESS_API_STARTUP_TIMEOUT_S", cls.startup_timeout_s)),
            max_upload_bytes=int(env.get("HARNESS_API_MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
            ingest_lock_stale_after_s=float(env.get("HARNESS_API_INGEST_LOCK_STALE_AFTER_S",
                                                    cls.ingest_lock_stale_after_s)),
            ingest_lock_renew_interval_s=float(env.get("HARNESS_API_INGEST_LOCK_RENEW_INTERVAL_S",
                                                       cls.ingest_lock_renew_interval_s)),
        )
