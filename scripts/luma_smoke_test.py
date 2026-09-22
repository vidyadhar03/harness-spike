"""OPT-IN live smoke test for the Luma Agents image adapter. Makes ONE paid generation.

It refuses to touch the network unless BOTH are true:
  - LUMA_AGENTS_API_KEY is set, and
  - --i-accept-charges is passed.
Run it once, deliberately, before wiring the frontend. It is not part of the test suite and nothing
imports it. See README.md ("Live smoke test") for what each step confirms.

    LUMA_AGENTS_API_KEY=... python scripts/luma_smoke_test.py --i-accept-charges [--model uni-1] [--refs 1]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

PROCEDURE = __doc__


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Luma smoke test (one paid generation)")
    ap.add_argument("--i-accept-charges", action="store_true")
    ap.add_argument("--model", default="uni-1")
    ap.add_argument("--refs", type=int, default=1, help="number of tiny inline reference images (1-3)")
    ap.add_argument("--max-wait", type=float, default=240.0)
    args = ap.parse_args()
    if not (os.environ.get("LUMA_AGENTS_API_KEY") and args.i_accept_charges):
        print(PROCEDURE)
        print("Not running: set LUMA_AGENTS_API_KEY and pass --i-accept-charges to spend money.")
        return 0

    from PIL import Image

    from harness.memory.files import validate_reference_image
    from harness.memory.imagegen import ImageInput, ImageRequest, estimate_request_bytes
    from harness.memory.luma import LumaImageProvider

    refs = []
    for color in ("#8a6f4d", "#4d6f8a", "#6f8a4d")[: max(1, min(args.refs, 3))]:
        buf = io.BytesIO()
        Image.new("RGB", (128, 128), color).save(buf, format="PNG")
        refs.append(ImageInput(buf.getvalue(), "image/png"))
    req = ImageRequest(
        prompt=("Reference image 1 is a colour swatch for the wall material. Create one image of a plain "
                "grey cube on a white studio floor, soft even lighting."),
        model=args.model, references=tuple(refs), aspect_ratio="1:1", output_format="png")

    p = LumaImageProvider(os.environ["LUMA_AGENTS_API_KEY"])
    print(f"estimated serialized request: {estimate_request_bytes(req)} bytes; refs={len(refs)}")
    gid = p.submit(req)                          # SubmissionRejected -> nothing created; Unknown -> check dashboard
    print(f"[1] submit accepted: generation id {gid}   (image_ref {{data, media_type}} was accepted)")

    deadline = time.time() + args.max_wait
    while True:
        job = p.get(gid)
        print(f"[2] state={job.state} kind={job.kind!r} model={job.model!r} created_at={job.created_at}")
        if job.state in ("completed", "failed") or time.time() > deadline:
            break
        time.sleep(3)
    if job.state != "completed":
        print(f"stopped in state {job.state}; failure_code={job.failure_code} reason={job.failure_reason}")
        print(f"The generation {gid} still exists at the provider - do not resubmit blindly.")
        return 1
    print(f"[3] provider reports kind/model/created_at -> attach verification is "
          f"{'POSSIBLE' if job.kind and job.model else 'NOT possible (fields missing)'}")
    img = p.download(job.output_urls[0])
    w, h = validate_reference_image(img.data, img.mime_type)
    print(f"[4] downloaded {len(img.data)} bytes, {img.mime_type}, {w}x{h} px; passed our import validation")
    print("smoke test OK - record the outputs above in the PR/notes; then delete nothing (the output expires on its own).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
