# video-harness — memory layer spike

Deterministic pipeline. Python controls the flow; Gemini is called at fixed points.
No API, no queue, no UI, no agent loop — that is deliberate for the spike.

## One-time: local environment

Requires Python 3.11+ (3.12 used). macOS system Python 3.9 will not work.

```bash
cd ~/video-harness
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pytest -q                      # expect all tests passing, no cloud needed
```

## One-time: GCP (project `film-harness`)

Needs a billing account attached (Blaze plan) — Vertex AI and Cloud Storage both require it.

```bash
gcloud auth login
gcloud config set project film-harness

# billing must show billingEnabled: true
gcloud billing projects describe film-harness

gcloud services enable firestore.googleapis.com storage.googleapis.com aiplatform.googleapis.com

gcloud firestore databases list        # skip create if one already exists
gcloud firestore databases create --location=asia-south1 --type=firestore-native

gcloud storage buckets create gs://film-harness-memory \
  --location=asia-south1 --uniform-bucket-level-access

# ADC is what the Python client libraries read — separate from `gcloud auth login`
gcloud auth application-default login
gcloud auth application-default set-quota-project film-harness
```

One manual edit: put a real contact address in `USER_AGENT` in
`harness/memory/wikimedia.py` (Wikimedia's API policy requires it).
Only needed before running `references`.

## Every new terminal

```bash
cd ~/video-harness
source .venv/bin/activate

export GCP_PROJECT=film-harness
export MEMORY_BUCKET=film-harness-memory
export GCP_LOCATION=global
export PID=prj_xxxxxxxxxxxx        # from `project create`
```

Optional overrides:

```bash
export MEMORY_MODEL=gemini-3.1-pro-preview        # extraction
export MEMORY_MODEL_FAST=gemini-3.1-flash-lite    # classification
export MEMORY_MAX_OUTPUT_TOKENS=65536             # lower it to force split-and-retry
export MEMORY_TEMPERATURE=1.0                     # unset = model default
```

## Health check (run after any auth or model change)

```bash
python - <<'EOF'
from harness.memory.config import Settings
from harness.memory.gcp import GeminiLLM
from harness.memory.ports import Text
from harness.memory.schemas import ClassifyOut
llm = GeminiLLM(Settings.from_env())
part = [Text("INT. PRASAD'S HOUSE - NIGHT\nPrasad enters.")]
print("flash:", llm.generate(system="Classify this file.", parts=part, schema=ClassifyOut, fast=True))
print("pro:  ", llm.generate(system="Classify this file.", parts=part, schema=ClassifyOut))
EOF
```

Covers ADC, both model ids, the inlined JSON schema and the output token cap.

## Dump layout

Folder names are passed to the model as evidence when matching images to locations,
so structure the dump:

```
~/dehleez_dump/
  scripts/Dehleez_Episode_1_Screenplay_Draft.pdf
  recce/devgram_well/IMG_0012.HEIC
  lookbook/dehleez_lookbook.pdf
  notes/director_notes.md
```

## Running

```bash
# create a project (prints the prj_ id)
export PID=$(harness-memory project create "Dehleez")
echo $PID

# register + digest (a folder or individual files)
harness-memory drop $PID ~/dehleez_dump
harness-memory drop $PID ~/dehleez_dump --no-ingest   # register only

# digest sources already registered
harness-memory ingest $PID
harness-memory ingest $PID --source <sha256>
harness-memory ingest $PID --force                    # re-digest, replacing proposed notes

# read
harness-memory status $PID
harness-memory index $PID
harness-memory context $PID "Tree Temple"
harness-memory context $PID "Scene 7"
harness-memory context $PID project
harness-memory context $PID "Tree Temple" --confirmed-only
harness-memory context $PID "Tree Temple" -o CONTEXT.md

# export every location and scene as markdown
harness-memory export $PID ./export_v1
harness-memory export $PID ./export_v1_confirmed --confirmed-only

# review: decisions are recorded against a specific revision of the assertion
harness-memory confirm $PID note_xxxx --by vd
harness-memory reject $PID note_xxxx --reason wrong_scope --by vd
harness-memory reject $PID note_xxxx --duplicate-of note_yyyy --by vd

# containment: only CONFIRMED parents inherit notes downward
harness-memory set-parent $PID "Market Square" "Devgram" --by vd
harness-memory confirm-parent $PID "Market Square" --by vd
harness-memory confirm-parent $PID "Orchards" --reject --by vd

# a new draft of an existing document
harness-memory drop $PID ~/dehleez_dump/scripts/Ep1_draft2.pdf \
  --supersedes <sha256-of-draft-1> --revision "Draft 2"

# merge a duplicate entity into the one you keep (source folds into target)
harness-memory merge $PID "Approach Road" "Village Road" --dry-run
harness-memory merge $PID "Approach Road" "Village Road"

# real-world visual references (per location)
harness-memory references $PID "Tree Temple" --terms-only  # vocabulary + verification only, seconds
harness-memory references $PID "Tree Temple" --dry-run     # full run, writes nothing (slow)
harness-memory references $PID "Tree Temple"
harness-memory references $PID "Tree Temple" --per-term 6 --max-images 32
```

## Running the API server

Thin FastAPI layer over `harness.memory.*` for `location-studio` - see
`harness/api/` and `location-studio/API_CONTRACT.md` for the endpoint contract. It
reuses the same `Settings`/adapters as the CLI (`harness/memory/factory.py`); the
`GCP_PROJECT`/`MEMORY_BUCKET`/etc. env vars from "Every new terminal" above apply here
too.

```bash
pip install -e ".[dev,api]"     # installs fastapi/uvicorn/httpx on top of the CLI deps
harness-api                     # binds 127.0.0.1:8000 by default
```

Optional overrides:

```bash
export HARNESS_API_HOST=127.0.0.1          # do not change to 0.0.0.0 without adding real
                                            # auth first - see below
export HARNESS_API_PORT=8000
export HARNESS_API_ALLOWED_ORIGINS=http://localhost:3000     # comma-separated; CORS
export HARNESS_API_ALLOWED_HOSTS=127.0.0.1,localhost         # comma-separated; Host header
export HARNESS_API_MAX_CONCURRENT_JOBS=2   # bounds concurrent references/ingest runs
export HARNESS_API_STARTUP_TIMEOUT_S=15    # bounds the one-time startup Firestore check;
                                            # see "If startup hangs" below before raising this
export HARNESS_API_MAX_UPLOAD_BYTES=52428800   # 50MB; POST /projects/{id}/sources limit,
                                                # enforced while receiving, not after
export HARNESS_API_INGEST_LOCK_STALE_AFTER_S=3600     # matches ingest.LOCK_STALE_AFTER_S -
                                                       # same value the CLI uses for the lock
                                                       # they share; see "shares the CLI's
                                                       # own lock" below before changing this
export HARNESS_API_INGEST_LOCK_RENEW_INTERVAL_S=900   # matches ingest.LOCK_RENEW_INTERVAL_S
```

There is deliberately no execution timeout on a references-pipeline run. It runs on a
real OS thread that Python cannot force-cancel; a timeout on the *await* would let the
job's lock and concurrency slot free (permitting a second concurrent run, or reporting a
false "failed") while the abandoned thread could still be mid-write. A run either
completes or the process is restarted - see `recover_orphaned_jobs` below - there is no
third option. The frontend's own poll loop already gives up waiting independently
(~12 minutes) without needing the server to time anything out.

Point `location-studio`'s `NEXT_PUBLIC_HARNESS_API_URL` at `http://127.0.0.1:8000` and
rebuild.

**Access boundary.** There is no user-auth model yet. The only real protection is
network-level: `harness-api` binds to `127.0.0.1` by default, `TrustedHostMiddleware`
rejects requests with an unrecognized `Host` header, and every mutation (`POST
/projects`, `POST .../sources`, `POST .../ingest`, `POST .../review`, `POST .../jobs`)
actively rejects a request whose `Origin` header is set but not in
`HARNESS_API_ALLOWED_ORIGINS` (`harness/api/deps.py:require_allowed_origin`,
403) - this check exists *in addition to* `CORSMiddleware` because CORS by itself is a
browser-enforced convention: for an actual (non-preflight) request, Starlette's
`CORSMiddleware` lets the request reach the route regardless of Origin and only adds or
omits the response header, relying on the browser to discard the response - by then the
mutation has already happened server-side. `TrustedHostMiddleware` doesn't cover this
gap either, since `Host` reflects this server (the request's target), not the page that
issued the request. A request with no `Origin` header at all (curl, direct local
testing) is still allowed through - non-browser access is bounded by the bind address
and `TrustedHostMiddleware` instead. None of this is authentication. Do not bind to
`0.0.0.0`, put this behind a public reverse proxy, or otherwise expose it beyond your
own machine/network without adding real authentication first.

**Single-process, localhost-only job execution.** References-pipeline runs execute as
background asyncio tasks in the same process (bounded by `HARNESS_API_MAX_CONCURRENT_JOBS`),
not a separate worker/queue. Job and per-location run-lock bookkeeping live in Firestore
(`projects/{id}/jobs`, `projects/{id}/job_locks`) so a job's outcome is inspectable after
a restart, but nothing resumes execution - on startup, `harness-api` marks every
job it finds still `queued`/`running` as `failed` ("interrupted: server restarted") and
drops its lock, since single-process means anything left in that state truly is
orphaned, regardless of age. Restarting is always safe to do; in-flight runs are simply
lost and need re-triggering.

**Ingestion jobs (`POST /projects/{id}/ingest`) share the CLI's own lock, not a second
one.** Unlike references jobs, ingestion can also be started from the CLI
(`harness-memory drop`/`ingest`) on the same project, entirely outside the API's
knowledge - so the API's `job_locks` mechanism above is deliberately *not* reused here;
it would let the CLI and the API each believe they're the sole writer and run
`ingest_source` on the same project concurrently, exactly what `ingest.ingest_lock`
(`Store.acquire_lock`/`release_lock`) exists to prevent. `JobRunner.trigger_ingest`
calls that same lock directly; a hit from the other side (CLI or a stale acquisition)
surfaces as `409`, not a guessed job id. Three related fixes shipped alongside this:

- `Store.acquire_lock`'s staleness check (`LOCK_STALE_AFTER_S`, 1 hour) is purely
  time-based and does not know whether its holder is still genuinely working. Renewing
  only *between* sources would still leave two periods exposed: waiting for a
  concurrency slot before any source has started, and a single source that itself runs
  long (many chunked extraction passes). `harness.memory.ingest.LockRenewer` closes
  both - a background thread that renews on a **fixed timer** (`LOCK_RENEW_INTERVAL_S`,
  15 min by default - `HARNESS_API_INGEST_LOCK_RENEW_INTERVAL_S` for the API side)
  independent of whatever phase the main work is in, started the moment the lock is
  acquired and stopped when the run ends (success, failure, or an early abort - always,
  via `finally`). Both the CLI's `drop`/`ingest` loops and the API's `JobRunner._run_ingest`
  use it, one implementation. If a renewal ever fails (`Store.renew_lock` returns
  `False` - someone else now holds the lock), the renewer sets `.lost` and stops; the
  caller checks `.lost` before starting each *next* source and stops there rather than
  continuing without ownership - the job/run ends up `"failed"` naming the lost
  ownership, never silently `"succeeded"`. What this cannot do: force-cancel a source
  that's already mid-flight when ownership is lost (the same limitation as running
  ingestion/references jobs with no execution timeout - a thread doing real work can't
  be interrupted); that source's own write, if any, has already happened by the time
  the loss is noticed. This shrinks the exposed window from "an entire multi-source
  batch" to "at most one in-flight source," it does not erase it entirely.
- `MemoryStore.acquire_lock`'s token used to be derived from the holder string alone
  (`f"lock_{holder}"`), so two different acquisitions by the same default holder (e.g.
  the CLI's `"cli"`) could produce identical tokens - harmless for the CLI's own
  single-lock-at-a-time usage, but a real hazard for the token-based restart recovery
  below, which trusts that a remembered token uniquely identifies *one* acquisition.
  Fixed to generate a unique token per call, matching `FirestoreStore` already.
- On restart, `run_startup_checks` releases an orphaned *ingest* job's lock using that
  job's own persisted token (`Store.release_lock`, which already refuses to delete a
  lock whose current token doesn't match) - never a blanket clear. A concurrently
  (or subsequently) running CLI ingest holds a different token and is left alone.

**If startup hangs at "Waiting for application startup."** - this happened once in
practice: `harness-api` sat there indefinitely and the frontend got
`ERR_CONNECTION_REFUSED`. The cause was an ADC credential needing reauthentication -
`google-auth`'s own token-refresh logic retried internally for minutes (observed:
~300s) before ever raising, and that hang happens *before* any Firestore RPC-level
timeout takes effect, so passing a timeout to the Firestore call itself does not help.
This is now bounded (`ApiSettings.startup_timeout_s`, default 15s - see
`harness.api.main.run_startup_checks`) and logged step by step. If it happens again:

```
INFO harness.api.main: startup: recovering orphaned jobs/locks (bounded to 15s)...
INFO harness.api.jobs: recover_orphaned_jobs: querying collection_group('jobs') for queued/running...
ERROR harness.api.main: startup: Firestore did not respond within 15s (elapsed 15.0s). This
almost always means the ADC credential needs reauthentication ... Run `gcloud auth
application-default login`, then restart harness-api.
```
...and the process now actually exits (rather than merely logging and hanging anyway -
see `run_startup_checks`'s docstring for why a plain `raise` wouldn't be enough here).
Run the command the log gives you, then restart `harness-api`. A credential that fails
*fast* rather than hanging (e.g. no ADC file at all) gets its own clear message the same
way, without the 15s wait. These logs need `logging.basicConfig` to be visible, which
`run()` sets up - if you're driving `create_app()`/uvicorn some other way, configure
logging yourself or you won't see any of this.

If you instead get a normal HTTP error (e.g. `address already in use`), that's not this
issue - check for a stale `harness-api` process still holding the port
(`lsof -i :8000`) before assuming it's a credentials problem.

**Verify locally without touching real GCP or Gemini:**

```bash
pytest tests/test_api.py -q     # uses MemoryStore/MemoryBlobs/MemoryImages and fake
                                 # LLMs throughout - no network, no credentials needed
```

**Known verification gap: the Firestore transaction paths are untested against real or
emulated Firestore** *by the automated suite* - `tests/test_api.py`'s startup-diagnostics
tests (`test_run_startup_checks_*`) cover the bounded-timeout/exception-translation logic
with fakes standing in for a hanging or broken Firestore call, which is the part that
was actually broken; they do not exercise `FirestoreJobStore`/`FirestoreStore` themselves
against a real backend. That has been done manually once, successfully (startup
completed, `GET /projects` returned real project data against the `film-harness`
project), but isn't repeated by CI/`pytest`. `tests/test_api.py` and
`tests/test_curate.py` exercise every API route and every JobRunner/review code path,
but always through `MemoryStore` and `InMemoryJobStore` - in-process dict equivalents,
not the real adapters. Two pieces of
code have real Firestore-specific behavior (transactions, `collection_group` queries)
that only run under `FirestoreStore`/`FirestoreJobStore` and are therefore **not**
covered by anything in this repo's test suite:

- `FirestoreStore.put_note_if_current` (`harness/memory/gcp.py`) - the compare-and-swap
  transaction backing review conflicts (409s).
- `FirestoreStore.renew_lock` (`harness/memory/gcp.py`) - the ingest-lock heartbeat
  transaction; only `MemoryStore`'s equivalent is exercised (`tests/test_api.py`'s
  `test_renew_lock_prevents_a_still_running_holder_from_losing_its_lock` and neighbors).
- `FirestoreJobStore` in full (`harness/api/jobs.py`) - `create_job`'s claim
  transaction, `release_lock_if_owner`'s transaction, `get_active_project_job`'s query,
  and `recover_orphaned_jobs`'s `collection_group("jobs")`/`collection_group("job_locks")`
  queries. Firestore typically requires a composite index for a `collection_group` query
  combined with a `where` filter - `recover_orphaned_jobs`/`get_active_project_job` may
  throw `FailedPrecondition` with an index-creation link the first time either runs
  against a real project, until that index exists.

Before relying on either in anything beyond local/manual use, run them against
`FIRESTORE_EMULATOR_HOST` (or the real `film-harness` project) at least once - in
particular, trigger `create_job` from two genuinely concurrent requests and confirm only
one wins, and let `recover_orphaned_jobs` run once to surface any missing index.

## Measuring a prompt change

Prompt edits change `digest_version`, so a re-run re-digests automatically.

```bash
harness-memory export $PID ./export_v1        # baseline BEFORE editing prompts
# edit harness/memory/prompts/*.md
harness-memory ingest $PID --force
harness-memory export $PID ./export_v2
diff -ru export_v1 export_v2 | less
```

Confirmed and rejected notes survive a re-digest; only proposed ones are replaced.

## What to watch in the output

- `unowned_notes: N` — notes the model returned with no owner. Should be ~0.
- `slugline not found` — the text layer extracted badly; chunking fell back to page numbers.
- `output truncated; split into N` — split-and-retry fired.
- `dropped unscoped note` — note had no location, scene, or project-wide scope.
- Locations with 0 notes, or duplicate locations that need merging (`harness-memory merge`).
- `inside X (proposed)` in the index — containment waiting for `confirm-parent`.
- "Some notes come from superseded drafts" in a context pack — reconcile after a new draft.

## Wiping spike data

```bash
gcloud firestore databases delete "(default)" --database="(default)"   # nuclear
gcloud storage rm -r gs://film-harness-memory/projects/$PID
```
