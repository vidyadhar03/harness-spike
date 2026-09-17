"""Construction of Store/Blobs/LLM/Images adapters from Settings.

Shared by the CLI and the API server so both build the exact same wiring.
"""
from __future__ import annotations

from .config import Settings
from .ingest import Ctx
from .references import RefCtx


def build_store(s: Settings):
    from .gcp import FirestoreStore

    return FirestoreStore(s.gcp_project, s.firestore_database)


def build_blobs(s: Settings):
    from .gcp import GCSBlobs

    return GCSBlobs(s.gcp_project)


def build_images():
    from .wikimedia import WikimediaImages

    return WikimediaImages()


def build_ctx(s: Settings) -> Ctx:
    from .gcp import GeminiLLM

    return Ctx(build_store(s), build_blobs(s), GeminiLLM(s), s)


def build_ref_ctx(s: Settings) -> RefCtx:
    from .gcp import GeminiLLM

    return RefCtx(build_store(s), build_blobs(s), GeminiLLM(s), build_images(), s)


def build_firestore_client(s: Settings):
    """A plain Firestore client, for API-owned collections outside the memory Store
    abstraction (e.g. job bookkeeping). Independent of build_store's client."""
    from google.cloud import firestore

    return firestore.Client(project=s.gcp_project, database=s.firestore_database)
