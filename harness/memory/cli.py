"""harness-memory CLI.

  harness-memory project create "Dehleez"
  harness-memory drop <project_id> ./dump [more paths...] [--no-ingest]
  harness-memory ingest <project_id> [--source SHA256] [--force]
  harness-memory status <project_id>
  harness-memory index <project_id>
  harness-memory context <project_id> "Devgram well" [--confirmed-only] [-o CONTEXT.md]
  harness-memory export <project_id> ./context_export [--confirmed-only]
  harness-memory references <project_id> "Devgram well" [--terms-only | --dry-run]
  harness-memory caption-eval capture <project_id> "Devgram" -o evals/devgram_candidates.json [--include "File title"]
  harness-memory caption-eval run evals/devgram_candidates.json [-n 3] [--no-reasons] [-o REPORT.md]
  harness-memory merge <project_id> "Approach Road" "Village Road" [--dry-run]
  harness-memory confirm <project_id> <note_id> [--by NAME]
  harness-memory reject <project_id> <note_id> --reason wrong_scope [--duplicate-of NOTE_ID]
  harness-memory set-parent <project_id> "Market Square" "Devgram" [--by NAME]
  harness-memory confirm-parent <project_id> "Market Square" [--by NAME]
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from .config import Settings
from .ingest import Ctx, IngestReport, ingest_lock, ingest_source, register_file
from .models import Location, Project, Scene
from .curate import merge_entities, review_containment, review_note
from .references import RefCtx, ReferenceReport, suggest_references
from .retrieval import export_context, get_context, render_context_md, render_index_md


def build_store(s: Settings):
    from .gcp import FirestoreStore

    return FirestoreStore(s.gcp_project, s.firestore_database)


def build_ctx(s: Settings) -> Ctx:
    from .gcp import GCSBlobs, GeminiLLM

    return Ctx(build_store(s), GCSBlobs(s.gcp_project), GeminiLLM(s), s)


def build_ref_ctx(s: Settings) -> RefCtx:
    from .gcp import GCSBlobs, GeminiLLM
    from .wikimedia import WikimediaImages

    return RefCtx(build_store(s), GCSBlobs(s.gcp_project), GeminiLLM(s), WikimediaImages(), s)


def iter_files(paths: list[str]):
    """Yields (path, name). Folder structure is kept in the name: it is evidence for the model."""
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(p)
                if f.is_file() and not any(part.startswith(".") for part in rel.parts):
                    yield f, f.relative_to(p.parent).as_posix()
        elif p.is_file():
            yield p, p.name
        else:
            print(f"skip: {raw} not found", file=sys.stderr)


def print_report(r: IngestReport) -> None:
    line = f"{r.status:<11} {r.filename}"
    if r.status == "digested":
        line += (f"  [{r.doc_type}] +{r.notes_written} notes, +{r.entities_created} entities, "
                 f"{r.notes_replaced} replaced")
    print(line)
    for name, count in sorted(r.warn_counts.items()):
        print(f"    {name}: {count}")
    if r.error:
        print(f"    error: {r.error}")
    for w in r.warnings[:10]:
        print(f"    warn: {w}")
    if len(r.warnings) > 10:
        print(f"    ... {len(r.warnings) - 10} more warnings")


def print_references(r: ReferenceReport) -> None:
    line = f"{r.location}: {len(r.terms_verified)}/{r.terms_proposed} terms verified"
    if r.images_found or r.images_kept:
        line += (f", {r.images_kept}/{r.images_found} images kept"
                 + (f" ({r.images_uncaptioned} judged irrelevant)" if r.images_uncaptioned else ""))
    line += f", {r.notes_written} notes ({r.notes_replaced} replaced)"
    print(line)
    if r.terms_verified:
        print("  terms:   " + ", ".join(r.terms_verified))
    if r.terms_dropped:
        print("  dropped: " + ", ".join(r.terms_dropped))
    if r.facets:
        # how many images could stand in for the place, vs only share its terrain or stonework
        print("  facets:  " + ", ".join(f"{v} {k}" for k, v in sorted(r.facets.items())))
    for name, count in r.directions:
        print(f"  · {name} ({count} images)")
    for title, origin in r.kept:
        # whether region scoping earns its keep: which searches the surviving images came from
        print(f"    kept: {title[:70]}  <- {origin}")
    for w in r.warnings[:10]:
        print(f"    warn: {w}")


def caption_eval_cmd(args, settings: Settings) -> int:
    from . import caption_eval as ce

    if args.action == "capture":
        ref_ctx = build_ref_ctx(settings)
        if ref_ctx.store.get_project(args.project_id) is None:
            print(f"project {args.project_id} not found", file=sys.stderr)
            return 1
        try:
            fx = ce.capture(ref_ctx, args.project_id, args.scope, include=args.include,
                            per_term=args.per_term, max_images=args.max_images)
        except (LookupError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        ce.save_fixture(args.out, fx)
        print(f"wrote {args.out}: {len(fx.hits)} candidates for {fx.location}"
              + (f", {len(fx.included)} included by hand" if fx.included else ""))
        return 0

    from .gcp import GeminiLLM
    from .wikimedia import WikimediaImages

    fx = ce.load_fixture(args.fixture)
    report, warnings = ce.run_eval(GeminiLLM(settings), ce.CachedImages(WikimediaImages(), args.cache), fx,
                                   n=args.n, reasons=not args.no_reasons)
    text = (f"fixture: {args.fixture} ({len(fx.hits)} candidates, captured {fx.captured_at})\n"
            f"caption prompt: {ce.prompt_fingerprint()}  model: {settings.model}  "
            f"temperature: {settings.temperature if settings.temperature is not None else 'model default'}\n\n"
            + ce.render(report)
            + "".join(f"\nwarn: {w}" for w in warnings))
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness-memory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("project").add_subparsers(dest="action", required=True).add_parser("create")
    pc.add_argument("name")
    d = sub.add_parser("drop")
    d.add_argument("project_id")
    d.add_argument("paths", nargs="+")
    d.add_argument("--no-ingest", action="store_true")
    d.add_argument("--supersedes", help="sha256 of the draft this file replaces")
    d.add_argument("--revision", help='label for this draft, e.g. "Draft 3"')
    i = sub.add_parser("ingest")
    i.add_argument("project_id")
    i.add_argument("--source")
    i.add_argument("--force", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("project_id")
    ix = sub.add_parser("index")
    ix.add_argument("project_id")
    cx = sub.add_parser("context")
    cx.add_argument("project_id")
    cx.add_argument("scope", help='entity id, location name or alias, scene number, or "project"')
    cx.add_argument("--confirmed-only", action="store_true")
    cx.add_argument("-o", "--out")
    rf = sub.add_parser("references")
    rf.add_argument("project_id")
    rf.add_argument("scope", help="location id, name, or alias")
    rf.add_argument("--per-term", type=int, default=6)
    rf.add_argument("--max-images", type=int, default=32)
    rf.add_argument("--dry-run", action="store_true", help="run the passes, write nothing")
    rf.add_argument("--terms-only", action="store_true",
                    help="stop after vocabulary verification; no image search or fetch")
    ce = sub.add_parser("caption-eval").add_subparsers(dest="action", required=True)
    cec = ce.add_parser("capture", help="freeze one run's candidate images and context pack as a fixture")
    cec.add_argument("project_id")
    cec.add_argument("scope", help="location id, name, or alias")
    cec.add_argument("-o", "--out", required=True)
    cec.add_argument("--include", action="append", default=[], help="Commons file title to append; repeatable")
    cec.add_argument("--per-term", type=int, default=6)
    cec.add_argument("--max-images", type=int, default=32)
    cer = ce.add_parser("run", help="run only the caption pass over a fixture, N times")
    cer.add_argument("fixture")
    cer.add_argument("-n", type=int, default=3, help="production runs; the flip count across them is the noise floor")
    cer.add_argument("--no-reasons", action="store_true", help="skip the separate diagnostic run")
    cer.add_argument("--cache", default=".cache/caption-eval", help="local image cache")
    cer.add_argument("-o", "--out", help="also write the report here")
    mg = sub.add_parser("merge")
    mg.add_argument("project_id")
    mg.add_argument("source", help="the duplicate to fold away (id, name, or alias)")
    mg.add_argument("target", help="the entity to keep")
    mg.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    cf = sub.add_parser("confirm")
    cf.add_argument("project_id")
    cf.add_argument("note_id")
    cf.add_argument("--by", default="user")
    rj = sub.add_parser("reject")
    rj.add_argument("project_id")
    rj.add_argument("note_id")
    rj.add_argument("--reason", choices=["false", "wrong_scope", "duplicate", "not_useful", "other"])
    rj.add_argument("--duplicate-of", help="the surviving note id; implies --reason duplicate")
    rj.add_argument("--by", default="user")
    sp = sub.add_parser("set-parent")
    sp.add_argument("project_id")
    sp.add_argument("child")
    sp.add_argument("parent")
    sp.add_argument("--by", default="user")
    cp = sub.add_parser("confirm-parent")
    cp.add_argument("project_id")
    cp.add_argument("child")
    cp.add_argument("--reject", action="store_true")
    cp.add_argument("--by", default="user")
    ex = sub.add_parser("export")
    ex.add_argument("project_id")
    ex.add_argument("out_dir")
    ex.add_argument("--confirmed-only", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if args.cmd == "caption-eval":
        return caption_eval_cmd(args, settings)
    store_only = args.cmd in ("status", "index", "context", "export", "merge",
                              "confirm", "reject", "set-parent", "confirm-parent")
    if args.cmd == "references":
        ctx = None
        ref_ctx = build_ref_ctx(settings)
        store = ref_ctx.store
    else:
        ctx = None if store_only else build_ctx(settings)
        store = build_store(settings) if store_only else ctx.store

    if args.cmd == "project":
        project = Project(name=args.name)
        store.put_project(project)
        print(project.id)
        return 0

    if store.get_project(args.project_id) is None:
        print(f"project {args.project_id} not found", file=sys.stderr)
        return 1

    if args.cmd == "index":
        print(render_index_md(store, args.project_id), end="")
        return 0

    if args.cmd == "context":
        try:
            pack = get_context(store, args.project_id, args.scope, include_proposed=not args.confirmed_only)
        except LookupError as exc:
            print(exc, file=sys.stderr)
            return 1
        text = render_context_md(pack)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text, end="")
        return 0

    if args.cmd == "references":
        try:
            report = suggest_references(ref_ctx, args.project_id, args.scope, per_term=args.per_term,
                                        max_images=args.max_images,
                                        dry_run=args.dry_run or args.terms_only,
                                        terms_only=args.terms_only)
        except (LookupError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        print_references(report)
        return 0

    if args.cmd == "merge":
        try:
            result = merge_entities(store, args.project_id, args.source, args.target, dry_run=args.dry_run)
        except (LookupError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        print(result.summary())
        return 0

    if args.cmd in ("confirm", "reject"):
        try:
            result = review_note(store, args.project_id, args.note_id,
                                 "confirmed" if args.cmd == "confirm" else "rejected",
                                 reviewer=args.by, reason=getattr(args, "reason", None),
                                 duplicate_of=getattr(args, "duplicate_of", None))
        except (LookupError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        print(result.summary())
        return 0

    if args.cmd in ("set-parent", "confirm-parent"):
        parent = getattr(args, "parent", None)
        decision = "rejected" if getattr(args, "reject", False) else "confirmed"
        try:
            loc = review_containment(store, args.project_id, args.child, decision,
                                     reviewer=args.by, parent_ref=parent)
        except (LookupError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        target = store.get_entity(args.project_id, loc.containment.parent_id)
        print(f"{loc.name} `{loc.id}` inside {target.name if target else loc.containment.parent_id} "
              f"· {loc.containment.status}")
        return 0

    if args.cmd == "export":
        paths = export_context(store, args.project_id, args.out_dir, include_proposed=not args.confirmed_only)
        print(f"wrote {len(paths)} files to {args.out_dir}")
        return 0

    if args.cmd == "drop":
        ids = []
        if args.supersedes and len(args.paths) != 1:
            print("--supersedes applies to a single file", file=sys.stderr)
            return 1
        for path, name in iter_files(args.paths):
            src, created = register_file(ctx, args.project_id, path.read_bytes(), name,
                                         revision_label=args.revision, supersedes=args.supersedes)
            print(f"{'new' if created else 'known':<5} {src.status:<11} {name}")
            ids.append(src.id)
        if not args.no_ingest:
            with ingest_lock(ctx, args.project_id):
                for sid in dict.fromkeys(ids):
                    print_report(ingest_source(ctx, args.project_id, sid))
        return 0

    if args.cmd == "ingest":
        sources = ([ctx.store.get_source(args.project_id, args.source)] if args.source
                   else ctx.store.list_sources(args.project_id))
        with ingest_lock(ctx, args.project_id):
            for src in sources:
                if src is not None:
                    print_report(ingest_source(ctx, args.project_id, src.id, force=args.force))
        return 0

    if args.cmd == "status":
        sources = store.list_sources(args.project_id)
        entities = store.list_entities(args.project_id)
        notes = store.list_notes(args.project_id)
        print(f"Sources ({len(sources)})")
        for s in sorted(sources, key=lambda s: s.filename):
            print(f"  {s.status:<11} {s.doc_type or '-':<9} {s.filename}" + (f"  ! {s.error}" if s.error else ""))
        per_scope = Counter(n.owner_id for n in notes if n.status != "rejected")
        print(f"\nLocations")
        for e in sorted((e for e in entities if isinstance(e, Location)), key=lambda e: e.name.lower()):
            print(f"  {e.id}  {e.status:<9} {per_scope[e.id]:>3} notes  {e.name}"
                  + (f"  (aka {', '.join(e.aliases)})" if e.aliases else ""))
        print(f"\nScenes: {sum(isinstance(e, Scene) for e in entities)}"
              f"   Project-wide notes: {per_scope['project']}"
              f"   Notes by status: {dict(Counter(n.status for n in notes))}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
