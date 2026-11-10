"""
Split batch recordings (several prompts read in one API call) into one clip per
prompt, using Whisper word timestamps instead of pauses.

synth_vieneu_api.py --batch N keeps every batch it could not cut cleanly by
pauses in <dataset>/_batches/batch_<first stem>_<N>.mp3. This script
transcribes each with word timestamps, aligns the words to the known prompt
text, and cuts at the quietest point between the last word of one prompt and
the first word of the next. Batches whose boundaries cannot be located
reliably are reported and left alone.

    uv sync --extra finetune
    uv run python finetune/split_batches.py
    # then: build_metadata.py, check_dataset.py (verifies every clip's words)
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from check_dataset import normalize   # noqa: E402

SR = 48_000
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
BATCH_RE = re.compile(r"^batch_(?P<prefix>.*?)(?P<num>\d{4})_(?P<n>\d+)\.mp3$")


def load(path: Path) -> np.ndarray:
    import librosa
    x, _ = librosa.load(str(path), sr=SR, mono=True)
    return x.astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> None:
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    tmp = path.with_suffix(".wav.part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    tmp.replace(path)


def frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    fr = len(x) // hop
    rms = np.sqrt(np.mean(x[: fr * hop].reshape(fr, hop) ** 2, axis=1)) + 1e-9
    return 20 * np.log10(rms / rms.max())


def trim(x: np.ndarray, silence_db: float = -40.0, pad_s: float = 0.12) -> np.ndarray:
    hop = SR // 100
    db = frame_db(x, hop)
    idx = np.flatnonzero(db >= silence_db)
    if len(idx) == 0:
        return x
    pad = int(pad_s * 100)
    s, e = max(0, idx[0] - pad), min(len(db), idx[-1] + 1 + pad)
    return x[s * hop: e * hop]


def quietest(x: np.ndarray, t0: float, t1: float) -> int:
    """Sample index of the quietest 10 ms frame between t0 and t1 seconds."""
    hop = SR // 100
    a, b = int(t0 * 100), max(int(t0 * 100) + 1, int(t1 * 100))
    db = frame_db(x, hop)
    b = min(b, len(db))
    if a >= b:
        return int(t0 * SR)
    return (a + int(np.argmin(db[a:b]))) * hop + hop // 2


def word_times(asr, x16: np.ndarray):
    out = asr({"raw": x16, "sampling_rate": 16000}, return_timestamps="word",
              generate_kwargs={"language": "vi", "task": "transcribe"})
    words = []
    for ch in out.get("chunks", []):
        s, e = ch["timestamp"]
        for w in normalize(ch["text"]):
            words.append((w, s, e if e is not None else s))
    return words


def cut_points(ref_sents: list[list[str]], hyp: list[tuple[str, float, float]]):
    """Times (seconds) between consecutive prompts, or None if unreliable."""
    ref = [w for s in ref_sents for w in s]
    sm = difflib.SequenceMatcher(None, ref, [w for w, _, _ in hyp], autojunk=False)
    r2h = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            r2h[blk.a + k] = blk.b + k
    cuts, pos = [], 0
    for sent in ref_sents[:-1]:
        last, first = pos + len(sent) - 1, pos + len(sent)
        pos += len(sent)
        # the boundary words themselves (or their direct neighbours) must be matched
        a = next((r2h[i] for i in (last, last - 1) if i in r2h), None)
        b = next((r2h[i] for i in (first, first + 1) if i in r2h), None)
        if a is None or b is None or b <= a:
            return None
        cuts.append((hyp[a][2], hyp[b][1]))
    return cuts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(ROOT / "finetune" / "dataset"))
    ap.add_argument("--prompts", default=str(ROOT / "finetune" / "prompts" / "cau_mau_vi.txt"))
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--force", action="store_true", help="re-split even if the clips already exist")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    import librosa
    import torch
    from transformers import pipeline

    ds = Path(args.dataset_dir)
    out_dir = ds / "raw_audio"
    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8-sig").splitlines()
               if l.strip() and not l.strip().startswith("#")]
    batches = sorted(p for p in (ds / "_batches").glob("batch_*.mp3") if BATCH_RE.match(p.name))
    jobs = []
    for p in batches:
        m = BATCH_RE.match(p.name)
        first, n, prefix = int(m["num"]), int(m["n"]), m["prefix"]
        lines = list(range(first, first + n))
        stems = [f"{prefix}{i:04d}" for i in lines]
        have = all(any((out_dir / f"{s}{e}").is_file() for e in AUDIO_EXTS) for s in stems)
        if have and not args.force:
            continue
        jobs.append((p, lines, stems))
    print(f"{len(batches)} batch recordings, {len(jobs)} to split")
    if not jobs:
        return

    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=args.model,
                   torch_dtype=torch.float16 if cuda else torch.float32, device=0 if cuda else -1)

    ok = bad = 0
    for p, lines, stems in jobs:
        x = load(p)
        hyp = word_times(asr, librosa.resample(x, orig_sr=SR, target_sr=16000))
        ref_sents = [normalize(prompts[i - 1]) for i in lines]
        cuts = cut_points(ref_sents, hyp)
        if cuts is None:
            bad += 1
            print(f"!! {p.name}: could not locate every prompt boundary; left alone")
            continue
        idx = [0] + [quietest(x, a, b) for a, b in cuts] + [len(x)]
        durs = []
        for s, a, b in zip(stems, idx[:-1], idx[1:]):
            seg = trim(x[a:b])
            write_wav(out_dir / f"{s}.wav", seg)
            durs.append(round(len(seg) / SR, 2))
        ok += 1
        print(f"{p.name}: {', '.join(stems)}  durations {durs}")
    print(f"split {ok} batch(es), {bad} left alone")


if __name__ == "__main__":
    main()
