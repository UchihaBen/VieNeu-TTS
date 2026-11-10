"""
Build finetune/dataset/metadata.csv from a prompt list + one audio file per line.

Workflow (e.g. when the clips are synthesized by another TTS, or recorded by
reading the prompts aloud):

  1. Take the prompt file (default: finetune/prompts/cau_mau_vi.txt), one
     sentence per line.
  2. Produce ONE audio file per line, named by its 1-based line number padded
     to 4 digits: line 1 -> 0001.wav, line 2 -> 0002.wav, ... (wav/flac/mp3/
     ogg/m4a all fine). Put them in finetune/dataset/raw_audio/.
  3. Run:
         uv run python finetune/build_metadata.py
     It pairs each line with its file, checks durations (3-15 s, like
     prepare_dataset.py), and writes metadata.csv. Then run prepare_dataset.py.

  Print the expected file names:  uv run python finetune/build_metadata.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def read_prompts(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if "|" in text:
            sys.exit(f"prompt contains '|', which metadata.csv uses as separator: {text!r}")
        lines.append(text)
    return lines


def find_audio(audio_dir: Path, stem: str) -> Path | None:
    for ext in AUDIO_EXTS:
        for cand in (audio_dir / f"{stem}{ext}", audio_dir / f"{stem}{ext.upper()}"):
            if cand.is_file():
                return cand
    return None


def duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        pass
    try:   # stdlib fallback, PCM .wav only
        import wave
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(ROOT / "finetune" / "prompts" / "cau_mau_vi.txt"))
    ap.add_argument("--audio-dir", default=str(ROOT / "finetune" / "dataset" / "raw_audio"))
    ap.add_argument("--extra", nargs=2, action="append", default=[], metavar=("PROMPTS", "PREFIX"),
                    help="another prompt list whose clips are named <PREFIX>0001.*, e.g. "
                         "--extra finetune/prompts/danh_van_vi.txt dv_ (repeatable)")
    ap.add_argument("--out", default=str(ROOT / "finetune" / "dataset" / "metadata.csv"))
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--list", action="store_true", help="print 'file_name<TAB>text' for every prompt and exit")
    args = ap.parse_args()
    try:   # Vietnamese text on a Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    sets = [(Path(args.prompts), "")] + [(Path(p), pre) for p, pre in args.extra]
    items = []   # (file stem, text)
    for path, prefix in sets:
        items += [(f"{prefix}{i:04d}", t) for i, t in enumerate(read_prompts(path), 1)]
    if args.list:
        for stem, text in items:
            print(f"{stem}\t{text}")
        return

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        sys.exit(f"audio folder not found: {audio_dir}")

    rows, missing, too_short, too_long, unreadable = [], [], [], [], []
    total = 0.0
    for stem, text in items:
        path = find_audio(audio_dir, stem)
        if path is None:
            missing.append(stem)
            continue
        dur = duration_seconds(path)
        if dur is None:
            unreadable.append(path.name)
            continue
        if dur < args.min_sec:
            too_short.append(f"{path.name} ({dur:.1f}s)")
            continue
        if dur > args.max_sec:
            too_long.append(f"{path.name} ({dur:.1f}s)")
            continue
        rows.append(f"{path.name}|{text}")
        total += dur

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("file_name|text\n" + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    print(f"prompts: {len(items)}   usable clips: {len(rows)}   total: {total / 60:.1f} min")
    for label, items in (("missing audio", missing), (f"shorter than {args.min_sec:g}s", too_short),
                         (f"longer than {args.max_sec:g}s", too_long), ("unreadable", unreadable)):
        if items:
            shown = ", ".join(items[:15]) + (f" ... (+{len(items) - 15})" if len(items) > 15 else "")
            print(f"  skipped, {label}: {shown}")
    print(f"wrote {out}")
    if total < 10 * 60:
        print("  note: under 10 minutes of audio; the fine-tune README recommends 10-30 minutes.")


if __name__ == "__main__":
    main()
