"""
Add the multi-sentence batch recordings as extra training clips.

A model fine-tuned only on single sentences (3-6 s) can fail to stop at the end
of a longer chunk and repeat its last words. The batch recordings made by
synth_vieneu_api.py --batch (5 prompts read in one go, ~15-20 s, in
<dataset>/_batches/) are natural multi-sentence reads, so they teach it to
read and finish longer chunks.

A batch is used only when every prompt in it is in the base metadata (i.e.
none of its sentences was dropped by check_dataset.py). Each one is copied to
raw_audio/long_<first stem>_<n>.mp3 and written, with the 5 texts joined,
into a new metadata file next to the base rows:

    uv run python finetune/add_batch_clips.py --extra finetune/prompts/lop_hoc_vi.txt lh_
    uv run python finetune/prepare_dataset.py --metadata finetune/dataset/metadata_long.csv --min-sec 2.5 --max-sec 25
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_RE = re.compile(r"^batch_(?P<prefix>.*?)(?P<num>\d{4})_(?P<n>\d+)\.mp3$")


def read_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines()
            if l.strip() and not l.strip().startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(ROOT / "finetune" / "dataset"))
    ap.add_argument("--metadata", default=None, help="base rows (default: <dataset-dir>/metadata_final.csv)")
    ap.add_argument("--out", default=None, help="default: <dataset-dir>/metadata_long.csv")
    ap.add_argument("--prompts", default=str(ROOT / "finetune" / "prompts" / "cau_mau_vi.txt"))
    ap.add_argument("--extra", nargs=2, action="append", default=[], metavar=("PROMPTS", "PREFIX"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ds = Path(args.dataset_dir)
    base = Path(args.metadata) if args.metadata else ds / "metadata_final.csv"
    out = Path(args.out) if args.out else ds / "metadata_long.csv"
    rows = [l for l in base.read_text(encoding="utf-8").splitlines() if l.strip()]
    header, body = (rows[0], rows[1:]) if rows and rows[0].lower().startswith("file_name|") else ("file_name|text", rows)
    kept_stems = {Path(l.split("|", 1)[0].strip()).stem for l in body}
    sets = {"": read_prompts(Path(args.prompts)), **{pre: read_prompts(Path(p)) for p, pre in args.extra}}

    added, skipped = [], []
    for p in sorted((ds / "_batches").glob("batch_*.mp3")):
        m = BATCH_RE.match(p.name)
        if not m or m["prefix"] not in sets:
            continue
        first, n, prefix = int(m["num"]), int(m["n"]), m["prefix"]
        prompts = sets[prefix]
        lines = list(range(first, first + n))
        stems = [f"{prefix}{i:04d}" for i in lines]
        missing = [s for s in stems if s not in kept_stems]
        if missing or lines[-1] > len(prompts):
            skipped.append(f"{p.name} (dropped: {', '.join(missing)})")
            continue
        dst = ds / "raw_audio" / f"long_{prefix}{first:04d}_{n}.mp3"
        shutil.copyfile(p, dst)
        added.append(f"{dst.name}|{' '.join(prompts[i - 1] for i in lines)}")

    out.write_text("\n".join([header, *body, *added]) + "\n", encoding="utf-8")
    print(f"base rows: {len(body)}   long clips added: {len(added)}   skipped batches: {len(skipped)}")
    for s in skipped:
        print("  skip", s)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
