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

    def _rpc(self, method, params=None, id_=1, session=None, timeout=300):
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
            if "token" in low and any(w in low for w in ("không đủ", "hết", "insufficient", "exceed", "vượt", "quota", "hạn mức")):
                raise QuotaExhausted(text)
            raise RuntimeError(f"{tool}: {text}")
        return {"text": text, **(res.get("structuredContent") or {})}


def synth_one(client: Client, text: str, voice: str, speed: float | None, out: Path) -> dict:
    args = {"text": text, "voice": voice}
    if speed:
        args["speed"] = speed
    res = client.call("text_to_speech", args)
    deadline = time.time() + 600
    while res.get("status") != "completed" or not res.get("audio_url"):
        if res.get("status") in ("failed", "error", "cancelled"):
            raise RuntimeError(f"job {res.get('job_id')} {res.get('status')}: {res.get('text')}")
        if time.time() > deadline or not res.get("job_id"):
            raise RuntimeError(f"no audio for job {res.get('job_id')}: {res.get('text')}")
        time.sleep(3)
        res = client.call("get_speech_status", {"job_id": res["job_id"]})
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(res["audio_url"], timeout=120) as r:
        tmp.write_bytes(r.read())
    if tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded audio is suspiciously small")
    tmp.replace(out)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True, help="voice id exactly as list_voices shows it")
    ap.add_argument("--prompts", default=str(ROOT / "finetune" / "prompts" / "cau_mau_vi.txt"))
    ap.add_argument("--out-dir", default=str(ROOT / "finetune" / "dataset" / "raw_audio"))
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="synthesize at most N missing lines")
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
        stem = f"{i:04d}"
        if any((out_dir / f"{stem}{ext}").is_file() for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            continue
        todo.append((i, text, out_dir / f"{stem}.mp3"))
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(prompts)} prompts, {len(prompts) - len(todo) if args.limit is None else '?'} already done, "
          f"{len(todo)} to synthesize with voice {args.voice!r}")

    stop = threading.Event()
    log_path = out_dir.parent / "synth_log.jsonl"
    log_lock = threading.Lock()
    done = failed = 0
    last_balance = None

    def work(item):
        i, text, out = item
        if stop.is_set():
            return i, "skipped", None
        for attempt in range(3):
            try:
                res = synth_one(client, text, args.voice, args.speed, out)
                return i, "ok", res
            except QuotaExhausted as e:
                stop.set()
                return i, "quota", str(e)
            except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as e:
                if attempt == 2:
                    return i, "error", str(e)
                time.sleep(5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, it) for it in todo]
        for fut in as_completed(futures):
            i, status, info = fut.result()
            if status == "ok":
                done += 1
                last_balance = info.get("spendable_tokens", last_balance)
                print(f"[{done}/{len(todo)}] {i:04d}.mp3  {info.get('duration_seconds', '?')}s  "
                      f"tokens left: {last_balance}")
                with log_lock, open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"line": i, "job_id": info.get("job_id"),
                                        "duration": info.get("duration_seconds")}) + "\n")
            elif status == "quota":
                print(f"!! out of tokens at line {i:04d}: {info}")
            elif status == "error":
                failed += 1
                print(f"!! line {i:04d} failed: {info}")

    remaining = sum(1 for it in todo if not it[2].is_file())
    print(f"done: {done} new, {failed} failed, {remaining} still missing"
          + (f", tokens left: {last_balance}" if last_balance is not None else ""))
    if remaining:
        print("re-run the same command later (e.g. after the daily cap resets) to finish.")


if __name__ == "__main__":
    main()
