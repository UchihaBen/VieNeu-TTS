"""
Vietnamese speech-to-text for apps/openai_speech.py (POST /v1/audio/transcriptions).
====================================================================================
Model: g-group-ai-lab/gipformer1.5-68M-rnnt (MIT), a 68M Zipformer RNN-T run by
sherpa-onnx on the CPU, so it never competes with TTS for the GPU or its
scheduler. Measured on 4 CPU threads: a 5-10 s clip decodes in ~0.2-0.6 s.

    uv pip install sherpa-onnx sherpa-onnx-core   # >= 1.13 ships the native lib in -core

Long audio is cut in pauses into 12-22 s pieces and decoded one after another:
attention cost grows faster than the length (73 s whole: 16 s; in pieces: 5 s),
and batching the pieces was measured slower still.

Environment (read by apps/openai_speech.py):
    VIENEU_STT=1          0 = do not load speech-to-text
    VIENEU_STT_DIR=...    folder holding encoder.onnx, decoder.onnx, joiner.onnx,
                          tokens.txt (~100 MB; missing files are downloaded into it).
                          Unset = Hugging Face cache, like the TTS models.
    VIENEU_STT_THREADS=4  CPU threads for decoding
"""
from __future__ import annotations

import io
import os
import struct
import tempfile
import threading
from typing import Optional

import numpy as np

MODEL_ID = "gipformer-1.5"
REPO = "g-group-ai-lab/gipformer1.5-68M-rnnt"
SAMPLE_RATE = 16_000
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 120.0
# The suffix only tells librosa/ffmpeg what to expect; it is mapped to a constant,
# never taken from the client.
EXTS = {e: e for e in (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3",
                       ".webm", ".m4a", ".mp4", ".aac")}


class AudioError(ValueError):
    """The upload is not audio we can read, or breaks a limit (→ HTTP 400)."""


class STT:
    """A loaded recognizer plus its own lock: one decode at a time, independent of TTS."""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.lock = threading.Lock()

    def transcribe(self, x: np.ndarray) -> str:
        """16 kHz mono float32 -> sentence-cased text ("" for < 0.1 s)."""
        if len(x) < SAMPLE_RATE // 10:
            return ""
        pieces = []
        with self.lock:
            for seg in split_on_pauses(x):
                s = self.recognizer.create_stream()
                s.accept_waveform(SAMPLE_RATE, seg)
                self.recognizer.decode_stream(s)
                pieces.append(s.result.text.strip())
        return sentence_case(" ".join(p for p in pieces if p))


def load(model_dir: Optional[str] = None, threads: int = 4,
         decoding: str = "modified_beam_search") -> STT:
    import sherpa_onnx
    from huggingface_hub import hf_hub_download

    paths = {}
    for key, fn in (("encoder", "encoder.onnx"), ("decoder", "decoder.onnx"),
                    ("joiner", "joiner.onnx"), ("tokens", "tokens.txt")):
        path = os.path.join(model_dir, fn) if model_dir else None
        if path is None or not os.path.exists(path):
            path = hf_hub_download(REPO, fn, local_dir=model_dir)
        paths[key] = path
    return STT(sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=paths["encoder"], decoder=paths["decoder"], joiner=paths["joiner"],
        tokens=paths["tokens"], num_threads=threads, sample_rate=SAMPLE_RATE,
        feature_dim=80, decoding_method=decoding,
    ))


def _fix_stream_wav(data: bytes) -> bytes:
    """A streamed WAV (ours included) claims an unknown, huge length; trim the
    ``data`` chunk size to the bytes actually there so libsndfile reads it."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    pos = 12
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        if cid == b"data":
            avail = len(data) - pos - 8
            if size <= avail:
                return data
            size = avail & ~3            # whole 16-bit stereo / 32-bit frames
            b = bytearray(data[:pos + 8 + size])
            struct.pack_into("<I", b, pos + 4, size)
            struct.pack_into("<I", b, 4, len(b) - 8)
            return bytes(b)
        pos += 8 + size + (size & 1)
    return data


def decode_16k(data: bytes, ext: str = ".wav", max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Bytes of a common audio file -> mono float32 at 16 kHz. Raises AudioError."""
    import librosa
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(_fix_stream_wav(data)), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        # libsndfile cannot read webm/m4a/aac; librosa hands them to audioread (needs ffmpeg).
        fd, tmp = tempfile.mkstemp(suffix=EXTS.get(ext, ".wav"))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            x, sr = librosa.load(tmp, sr=None, mono=True)
        except Exception as e:   # noqa: BLE001
            raise AudioError("could not decode the audio; send WAV/FLAC/OGG/MP3 "
                             "(webm/m4a need ffmpeg on the server)") from e
        finally:
            os.unlink(tmp)
    if not sr or sr <= 0:
        raise AudioError("could not decode the audio: no sample rate")
    if len(x) / sr > max_seconds:
        raise AudioError(f"audio is {len(x) / sr:.0f} s long; the limit is {max_seconds:.0f} s")
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    if sr != SAMPLE_RATE and len(x):
        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(x, dtype=np.float32)


def split_on_pauses(x: np.ndarray, min_s: float = 12.0, max_s: float = 22.0,
                    frame_s: float = 0.1) -> list:
    """Cut audio longer than ``max_s`` at the quietest 100 ms frame between
    ``min_s`` and ``max_s`` after the previous cut, until the rest fits."""
    sr, n = SAMPLE_RATE, len(x)
    hop, out, pos = int(frame_s * sr), [], 0
    while n - pos > max_s * sr:
        lo, hi = pos + int(min_s * sr), pos + int(max_s * sr)
        frames = x[lo:hi][: (hi - lo) // hop * hop].reshape(-1, hop)
        cut = lo + int(np.argmin((frames ** 2).mean(axis=1))) * hop + hop // 2
        out.append(x[pos:cut])
        pos = cut
    out.append(x[pos:])
    return out


def sentence_case(text: str) -> str:
    """The model emits UPPERCASE without punctuation; lower it and capitalise the first letter."""
    text = text.lower()
    return text[:1].upper() + text[1:]
