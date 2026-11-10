"""
Synthesize the prompt list with a voice from the hosted VieNeu API (api.vieneu.io,
MCP over HTTP) to build a fine-tuning dataset: line N -> raw_audio/000N.mp3.

    export VIENEU_API_KEY=vn_sk_...            # never commit the key
    python finetune/synth_vieneu_api.py --voice "Ban Mai"

- Resumable: lines whose file already exists are skipped, so re-running after a
  quota stop (daily cap) only synthesizes what is missing.
- Stops cleanly when the account runs out of tokens.
- Spends the account's tokens (billed per character, min 50 per call). Check
  the provider's terms allow using the output for training.
Then run finetune/build_metadata.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.vieneu.io/mcp"


class QuotaExhausted(Exception):
    pass


class Client:
    def __init__(self, key: str):
        self.key = key

    def _rpc(self, method, params=None, id_=1, session=None, timeout=90):
        body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if id_ is not None:
            body["id"] = id_
        h = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
        if session:
            h["Mcp-Session-Id"] = session
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            sid = r.headers.get("Mcp-Session-Id")
            raw = r.read().decode("utf-8")
        msgs = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:")]
        if not msgs and raw.strip():
            msgs = [json.loads(raw)]
        return sid, (msgs[-1] if msgs else {})

    def call(self, tool: str, args: dict) -> dict:
        sid, _ = self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                          "clientInfo": {"name": "vieneu-dataset", "version": "1.0"}})
        self._rpc("notifications/initialized", id_=None, session=sid)
        _, msg = self._rpc("tools/call", {"name": tool, "arguments": args}, id_=2, session=sid)
        if "error" in msg:
            raise RuntimeError(f"{tool}: {msg['error']}")
        res = msg.get("result", {})
        text = " ".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if res.get("isError"):
            low = text.lower()
            # Out of tokens, or the plan's daily call cap ("50 API calls per day" on trials).
            if (any(w in low for w in ("hạn mức", "limit reached", "quota", "đạt trần"))
                    or ("token" in low and any(w in low for w in ("không đủ", "hết", "insufficient", "exceed")))):
                raise QuotaExhausted(text)
            raise RuntimeError(f"{tool}: {text}")
        return {"text": text, **(res.get("structuredContent") or {})}


def synth_one(client: Client, text: str, voice: str, speed: float | None, out: Path) -> dict:
    res = _synthesize(client, text, voice, speed)
    _download(res["audio_url"], out)
    return res


def _synthesize(client: Client, text: str, voice: str, speed: float | None) -> dict:
    args = {"text": text, "voice": voice}
    if speed:
        args["speed"] = speed
    res = client.call("text_to_speech", args)
    deadline = time.time() + 180
    while res.get("status") != "completed" or not res.get("audio_url"):
        if res.get("status") in ("failed", "error", "cancelled"):
            raise RuntimeError(f"job {res.get('job_id')} {res.get('status')}: {res.get('text')}")
        if time.time() > deadline or not res.get("job_id"):
            raise RuntimeError(f"no audio for job {res.get('job_id')}: {res.get('text')}")
        time.sleep(3)
        res = client.call("get_speech_status", {"job_id": res["job_id"]})
    return res


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r:
        tmp.write_bytes(r.read())
    if tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded audio is suspiciously small")
    tmp.replace(out)


# ── batch mode: several prompts per call, cut at the pauses between them ─────

SR = 48_000


def _ffmpeg() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("batch mode needs ffmpeg (apt install ffmpeg, or pip install imageio-ffmpeg)")


def _decode(path: Path):
    import subprocess
    import numpy as np
    raw = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _write_wav(path: Path, x) -> None:
    import wave
    import numpy as np
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    tmp = path.with_suffix(path.suffix + ".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    tmp.replace(path)


def split_at_pauses(x, n: int, char_lens: list[int], silence_db: float = -40.0, pad_s: float = 0.12):
    """Cut ``x`` into ``n`` utterances at its ``n-1`` longest interior pauses.

    Returns the list of segments, or None when the cut does not look right
    (too few pauses, or a segment's duration far off what its text length
    predicts) so the caller can retry the batch one prompt per call.
    """
    import numpy as np
    hop = SR // 100                                   # 10 ms frames
    frames = len(x) // hop
    if n == 1:
        bounds = [(0, frames)]
    else:
        rms = np.sqrt(np.mean(x[: frames * hop].reshape(frames, hop) ** 2, axis=1)) + 1e-9
        silent = 20 * np.log10(rms / rms.max()) < silence_db
        runs, start = [], None
        for i, s in enumerate(silent):
            if s and start is None:
                start = i
            elif not s and start is not None:
                runs.append((start, i))
                start = None
        interior = [r for r in runs if r[0] > 0]      # a trailing run never closes, so it is excluded
        if len(interior) < n - 1:
            return None
        picked = sorted(sorted(interior, key=lambda r: r[1] - r[0], reverse=True)[: n - 1])
        if (picked and min(e - s for s, e in picked) < 15):   # a sentence break should be >= 150 ms
            return None
        cuts = [0] + [(s + e) // 2 for s, e in picked] + [frames]
        bounds = list(zip(cuts[:-1], cuts[1:]))

    rms_all = np.sqrt(np.mean(x[: frames * hop].reshape(frames, hop) ** 2, axis=1)) + 1e-9
    voiced = 20 * np.log10(rms_all / rms_all.max()) >= silence_db
    pad = int(pad_s * 100)
    segs = []
    for a, b in bounds:
        idx = np.flatnonzero(voiced[a:b])
        if len(idx) == 0:
            return None
        s, e = max(a, a + idx[0] - pad), min(b, a + idx[-1] + 1 + pad)
        segs.append(x[s * hop: e * hop])
    if n > 1:
        ratios = np.array([len(sg) / SR / max(1, c) for sg, c in zip(segs, char_lens)])
        med = float(np.median(ratios))
        if np.any(ratios < 0.6 * med) or np.any(ratios > 1.6 * med):
            return None
    return segs


def synth_batch(client: Client, items: list, voice: str, speed: float | None, work_dir: Path) -> list:
    """items: [(line, text, out_path)]; writes out_path.with_suffix('.wav') per item."""
    first = items[0][0]
    res = _synthesize(client, " ".join(t for _, t, _ in items), voice, speed)
    batch_mp3 = work_dir / f"batch_{items[0][2].stem}_{len(items)}.mp3"
    _download(res["audio_url"], batch_mp3)
    segs = split_at_pauses(_decode(batch_mp3), len(items), [len(t) for _, t, _ in items])
    if segs is None:
        raise RuntimeError(f"could not cut batch starting at line {first} cleanly ({batch_mp3.name})")
    for (_, _, out), seg in zip(items, segs):
        _write_wav(out.with_suffix(".wav"), seg)
    return [round(len(s) / SR, 2) for s in segs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True, help="voice id exactly as list_voices shows it")
    ap.add_argument("--prompts", default=str(ROOT / "finetune" / "prompts" / "cau_mau_vi.txt"))
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "dataset" / "raw_audio"))
    ap.add_argument("--prefix", default="", help="file name prefix, e.g. dv_ -> dv_0001.mp3")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="synthesize at most N missing lines")
    ap.add_argument("--batch", type=int, default=1,
                    help="prompts per API call (cut apart at the pauses; saved as .wav). "
                         "Use 4-6 on plans with a daily call cap.")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    key = os.environ.get("VIENEU_API_KEY")
    if not key:
        sys.exit("set VIENEU_API_KEY first")
    client = Client(key)

    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8-sig").splitlines()
               if l.strip() and not l.strip().startswith("#")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for i, text in enumerate(prompts, 1):
        stem = f"{args.prefix}{i:04d}"
        if any((out_dir / f"{stem}{ext}").is_file() for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            continue
        todo.append((i, text, out_dir / f"{stem}.mp3"))
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(prompts)} prompts, {len(prompts) - len(todo) if args.limit is None else '?'} already done, "
          f"{len(todo)} to synthesize with voice {args.voice!r}")

    stop = threading.Event()
    log_path = out_dir.parent / "synth_log.jsonl"
    work_dir = out_dir.parent / "_batches"
    log_lock = threading.Lock()
    done = failed = 0
    last_balance = None
    size = max(1, args.batch)
    units = [todo[k:k + size] for k in range(0, len(todo), size)]
    if size > 1:
        print(f"batch mode: {len(units)} calls of up to {size} prompts, saved as .wav")

    def work(unit):
        if stop.is_set():
            return unit, "skipped", None
        for attempt in range(3):
            try:
                if len(unit) == 1 and size == 1:
                    i, text, out = unit[0]
                    res = synth_one(client, text, args.voice, args.speed, out)
                    return unit, "ok", {**res, "durations": [res.get("duration_seconds")]}
                return unit, "ok", {"durations": synth_batch(client, unit, args.voice, args.speed, work_dir)}
            except QuotaExhausted as e:
                stop.set()
                return unit, "quota", str(e)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 2:
                    return unit, "error", str(e)
                time.sleep(5 * (attempt + 1))
            except RuntimeError as e:      # failed job or unclean cut: retrying would spend another call
                return unit, "error", str(e)

    def name(i):
        return f"{args.prefix}{i:04d}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, u) for u in units]
        for fut in as_completed(futures):
            unit, status, info = fut.result()
            lines = ", ".join(name(i) for i, _, _ in unit)
            if status == "ok":
                done += len(unit)
                last_balance = info.get("spendable_tokens", last_balance)
                print(f"[{done}/{len(todo)}] {lines}  durations {info['durations']}"
                      + (f"  tokens left: {last_balance}" if last_balance is not None else ""))
                with log_lock, open(log_path, "a", encoding="utf-8") as f:
                    for (i, _, _), d in zip(unit, info["durations"]):
                        f.write(json.dumps({"prefix": args.prefix, "line": i, "duration": d,
                                            "batch": len(unit)}) + "\n")
            elif status == "quota":
                print(f"!! stopped at {lines}: plan limit / tokens: {info}")
            elif status == "error":
                failed += len(unit)
                print(f"!! {lines} failed: {info}")

    remaining = sum(1 for _, _, out in todo
                    if not out.is_file() and not out.with_suffix(".wav").is_file())
    print(f"done: {done} new, {failed} failed, {remaining} still missing"
          + (f", tokens left: {last_balance}" if last_balance is not None else ""))
    if remaining:
        print("re-run the same command later (e.g. after the daily cap resets) to finish.")


if __name__ == "__main__":
    main()
