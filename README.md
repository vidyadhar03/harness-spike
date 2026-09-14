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
