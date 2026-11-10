"""
Find clips that start or end with a stray fragment of speech.

A clip cut out of a longer recording can keep the first consonant of the next
sentence (or the last bit of the previous one) behind a short pause. Whisper
usually ignores such fragments, so check_dataset.py passes them, but training
on them teaches the model to say "a little more" after the text ends.

Flags a clip when its first or last voiced segment is shorter than
--max-frag seconds and separated from the rest by >= --min-gap seconds.

    uv run python finetune/check_edges.py                      # all clips in metadata.csv
    uv run python finetune/check_edges.py --metadata finetune/dataset/metadata_long.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def segments(x: np.ndarray, thr_db: float, join_gap: float):
    hop = SR // 100
    fr = len(x) // hop
    if fr == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(x[: fr * hop].reshape(fr, hop) ** 2, axis=1)) + 1e-9)
    db -= db.max()
    segs, st, last = [], None, -99
    for i, on in enumerate(db > thr_db):
        if on:
            if st is None:
                st = i
            elif i - last > join_gap * 100:
                segs.append((st / 100, (last + 1) / 100))
                st = i
            last = i
    if st is not None:
        segs.append((st / 100, (last + 1) / 100))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(ROOT / "finetune" / "dataset"))
    ap.add_argument("--metadata", default=None, help="default: <dataset-dir>/metadata.csv")
    ap.add_argument("--max-frag", type=float, default=0.25)
    ap.add_argument("--min-gap", type=float, default=0.08)
    ap.add_argument("--thr-db", type=float, default=-35.0)
    ap.add_argument("--write-clean", default=None, help="write the metadata without flagged clips here")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import librosa

    ds = Path(args.dataset_dir)
    meta = Path(args.metadata) if args.metadata else ds / "metadata.csv"
    lines = [l for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
    header, body = (lines[0], lines[1:]) if lines[0].lower().startswith("file_name|") else ("file_name|text", lines)

    stats = defaultdict(lambda: [0, 0, []])
    flagged = set()
    for line in body:
        name = line.split("|", 1)[0].strip()
        kind = "long (uncut)" if name.startswith("long_") else (
            "single call (uncut)" if name.endswith(".mp3") else "cut from a batch")
        x, _ = librosa.load(str(ds / "raw_audio" / name), sr=SR, mono=True)
        s = segments(x, args.thr_db, args.min_gap)
        tail = len(s) > 1 and s[-1][1] - s[-1][0] < args.max_frag and s[-1][0] - s[-2][1] >= args.min_gap
        head = len(s) > 1 and s[0][1] - s[0][0] < args.max_frag and s[1][0] - s[0][1] >= args.min_gap
        st = stats[kind]
        st[0] += 1
        if tail or head:
            st[1] += 1
            flagged.add(name)
            frag = s[-1] if tail else s[0]
            st[2].append(f"{name} ({'end' if tail else 'start'} {frag[1] - frag[0]:.2f}s)")
    for kind, (n, bad, ex) in stats.items():
        print(f"{kind:22} clips {n:4}   with a stray fragment {bad:3} ({bad / max(1, n):.0%})")
        for e in ex[:20]:
            print("     ", e)
    if args.write_clean:
        keep = [l for l in body if l.split("|", 1)[0].strip() not in flagged]
        Path(args.write_clean).write_text("\n".join([header, *keep]) + "\n", encoding="utf-8")
        print(f"wrote {args.write_clean} ({len(keep)} clips, {len(flagged)} dropped)")


if __name__ == "__main__":
    main()
