from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gcp_project: str
    bucket: str                              # bucket name, no gs:// prefix
    gcp_location: str = "global"
    firestore_database: str = "(default)"
    model: str = "gemini-3.1-pro-preview"    # extraction
    model_fast: str = "gemini-3.1-flash-lite"  # classification
    temperature: float | None = None         # None = model default; Gemini 3 recommends 1.0
    max_output_tokens: int = 65_536          # includes thinking tokens; lower it if a model rejects this
    max_retries: int = 4
    inline_limit_bytes: int = 15 * 1024 * 1024
    script_chunk_chars: int = 40_000         # text scripts: whole scenes packed up to this size
    scanned_script_chunk_pages: int = 8      # scanned scripts: whole scenes packed up to this many pages
    text_chunk_pages: int = 10
    visual_chunk_pages: int = 6
    text_chunk_chars: int = 40_000
    render_scale: float = 1.5                # ~1240px wide for A4

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        temp = env.get("MEMORY_TEMPERATURE")
        return cls(
            gcp_project=env["GCP_PROJECT"],
            bucket=env["MEMORY_BUCKET"],
            gcp_location=env.get("GCP_LOCATION", cls.gcp_location),
            firestore_database=env.get("FIRESTORE_DATABASE", cls.firestore_database),
            model=env.get("MEMORY_MODEL", cls.model),
            model_fast=env.get("MEMORY_MODEL_FAST", cls.model_fast),
            temperature=float(temp) if temp else None,
            max_output_tokens=int(env.get("MEMORY_MAX_OUTPUT_TOKENS", cls.max_output_tokens)),
        )
