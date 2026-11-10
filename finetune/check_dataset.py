"""
Check that every clip in a fine-tuning dataset actually says its text.

Transcribes each clip listed in metadata.csv with Whisper (GPU recommended) and
compares it with the expected text (word error rate after lower-casing and
stripping punctuation). Clips above --max-wer are flagged: listen to them,
then delete + re-synthesize, or drop them.

    uv sync --extra finetune
    uv run python finetune/check_dataset.py --dataset-dir finetune/dataset

Writes <dataset-dir>/asr_report.csv (all clips) and, with --write-clean,
<dataset-dir>/metadata_clean.csv without the flagged clips (pass it to
prepare_dataset.py with --metadata).

Whisper may write numbers as digits ("15") where the text spells them out
("mười lăm"); such clips are marked "digits" so they are not mistaken for
real misreads — just give them a listen.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def read_metadata(path: Path) -> list[tuple[str, str]]:
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or (ln == 1 and line.lower().startswith("file_name|")):
            continue
        name, _, text = line.partition("|")
        rows.append((name.strip(), text.split("|")[0].strip()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(ROOT / "finetune" / "dataset"))
    ap.add_argument("--metadata", default=None, help="default: <dataset-dir>/metadata.csv")
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--max-wer", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--write-clean", action="store_true")
    ap.add_argument("--exclude-prefix", action="append", default=[],
                    help="skip clips whose file name starts with this (they stay in the clean list), "
                         "e.g. dv_ for spelling drills Whisper cannot transcribe")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    import librosa
    import torch
    from transformers import pipeline

    ds = Path(args.dataset_dir)
    meta = Path(args.metadata) if args.metadata else ds / "metadata.csv"
    all_rows = read_metadata(meta)
    rows = [r for r in all_rows if not any(r[0].startswith(p) for p in args.exclude_prefix)]
    if len(rows) != len(all_rows):
        print(f"skipping {len(all_rows) - len(rows)} clips by prefix {args.exclude_prefix}")
    audio_dir = ds / "raw_audio"
    print(f"{len(rows)} clips in {meta}; model {args.model}")

    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=args.model,
                   torch_dtype=torch.float16 if cuda else torch.float32,
                   device=0 if cuda else -1)
    gen = {"language": "vi", "task": "transcribe"}

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        inputs = []
        for name, _ in batch:
            wav, _ = librosa.load(str(audio_dir / name), sr=16000, mono=True)
            inputs.append({"raw": wav, "sampling_rate": 16000})
        outs = asr(inputs, batch_size=args.batch_size, generate_kwargs=gen)
        for (name, text), out in zip(batch, outs):
            hyp = out["text"].strip()
            score = wer(normalize(text), normalize(hyp))
            if score <= args.max_wer:
                status = "ok"
            elif re.search(r"\d", hyp):
                status = "digits"
            else:
                status = "FLAG"
            results.append((name, score, status, text, hyp))
        print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}")

    report = ds / "asr_report.csv"
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "wer", "status", "expected", "heard"])
        for r in sorted(results, key=lambda r: -r[1]):
            w.writerow([r[0], f"{r[1]:.3f}", r[2], r[3], r[4]])

    flagged = [r for r in results if r[2] == "FLAG"]
    digits = [r for r in results if r[2] == "digits"]
    print(f"\nok: {len(results) - len(flagged) - len(digits)}   flagged: {len(flagged)}   "
          f"digits (listen): {len(digits)}   report: {report}")
    for name, score, status, text, hyp in sorted(flagged + digits, key=lambda r: -r[1]):
        print(f"\n[{status}] {name}  WER {score:.2f}\n  expected: {text}\n  heard:    {hyp}")

    if args.write_clean:
        bad = {r[0] for r in flagged}
        clean = ds / "metadata_clean.csv"
        with open(clean, "w", encoding="utf-8") as f:
            f.write("file_name|text\n")
            for name, text in all_rows:
                if name not in bad:
                    f.write(f"{name}|{text}\n")
        print(f"\nwrote {clean} ({len(all_rows) - len(bad)} clips)")


if __name__ == "__main__":
    main()
