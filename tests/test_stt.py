"""apps/stt.py and POST /v1/audio/transcriptions with a stand-in recognizer
(no sherpa-onnx or model download needed)."""
import io
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from apps import openai_speech as api
from apps import stt as stt_mod
from test_openai_speech import FakeTTS


class FakeRecognizer:
    def __init__(self):
        self.seg_lens = []

    def create_stream(self):
        rec = self

        class Stream:
            result = SimpleNamespace(text="")

            def accept_waveform(self, sr, x):
                assert sr == stt_mod.SAMPLE_RATE and x.dtype == np.float32
                rec.seg_lens.append(len(x))
                self.result = SimpleNamespace(text=" XIN CHÀO HÀ NỘI ")
        return Stream()

    def decode_stream(self, s):
        pass


def _wav(seconds=1.0, rate=24_000, channels=1) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    b = io.BytesIO()
    sf.write(b, x, rate, format="WAV", subtype="PCM_16")
    return b.getvalue()


@pytest.fixture
def rec(monkeypatch, tmp_path):
    for var in ("VIENEU_API_KEY", "VIENEU_MAX_STREAMS", "VIENEU_QUEUE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VIENEU_VOICES_FILE", str(tmp_path / "custom_voices.json"))
    monkeypatch.setattr(api, "Vieneu", lambda **kw: FakeTTS())
    monkeypatch.setattr(api, "ENGINE", api.Engine())
    r = FakeRecognizer()
    monkeypatch.setattr(stt_mod, "load", lambda **kw: stt_mod.STT(r))
    monkeypatch.setattr(api, "STT", None)
    return r


@pytest.fixture
def client(rec):
    with TestClient(api.app) as c:
        yield c


def _post(client, data, name="a.wav", **form):
    return client.post("/v1/audio/transcriptions", files={"file": (name, data)}, data=form)


def test_transcribes_to_sentence_case(client, rec):
    r = _post(client, _wav(rate=24_000, channels=2))
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "Xin chào hà nội"}
    assert rec.seg_lens == [16_000]            # resampled 24 kHz stereo -> 16 kHz mono


def test_response_formats(client):
    assert _post(client, _wav(), response_format="text").text == "Xin chào hà nội"
    v = _post(client, _wav(2.0), response_format="verbose_json", model="whisper-1", language="vi").json()
    assert v["text"] == "Xin chào hà nội" and v["duration"] == 2.0 and "inference_seconds" in v
    assert _post(client, _wav(), response_format="srt").status_code == 400


def test_bad_language_and_garbage_are_400(client):
    assert _post(client, _wav(), language="en").status_code == 400
    r = _post(client, b"not audio at all" * 100, name="x.webm")
    assert r.status_code == 400 and "decode" in r.json()["error"]["message"]


def test_limits(client, monkeypatch):
    assert _post(client, _wav(121, rate=8_000)).status_code == 400
    monkeypatch.setattr(stt_mod, "MAX_BYTES", 1000)
    assert _post(client, _wav()).status_code == 413


def test_streamed_wav_header_is_read(client, rec):
    # The header our own /v1/audio/speech sends: RIFF and data sizes 0xFFFFFFFF.
    b = bytearray(_wav(1.0, rate=16_000))
    struct.pack_into("<I", b, 4, 0xFFFFFFFF)
    struct.pack_into("<I", b, 40, 0xFFFFFFFF)
    assert _post(client, bytes(b)).status_code == 200
    assert rec.seg_lens == [16_000]


def test_tts_to_stt_round_trip(client, rec):
    wav = client.post("/v1/audio/speech", json={"input": "Xin chào.", "response_format": "wav"}).content
    assert _post(client, wav).status_code == 200
    assert rec.seg_lens == [1600]              # 4800 samples at 48 kHz -> 0.1 s at 16 kHz


def test_auth_health_models(client, monkeypatch):
    assert client.get("/health").json()["stt"] is True
    assert stt_mod.MODEL_ID in [m["id"] for m in client.get("/v1/models").json()["data"]]
    monkeypatch.setenv("VIENEU_API_KEY", "k")
    assert _post(client, _wav()).status_code == 401
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav())},
                    headers={"Authorization": "Bearer k"})
    assert r.status_code == 200


def test_load_failure_leaves_tts_up(rec, monkeypatch):
    def boom(**kw):
        raise ImportError("No module named 'sherpa_onnx'")
    monkeypatch.setattr(stt_mod, "load", boom)
    with TestClient(api.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["stt"] is False
        assert _post(c, _wav()).status_code == 503
        assert [m["id"] for m in c.get("/v1/models").json()["data"]] == [api.MODEL_ID]


def test_disabled_by_env(rec, monkeypatch):
    monkeypatch.setenv("VIENEU_STT", "0")
    with TestClient(api.app) as c:
        assert c.get("/health").json()["stt"] is False


def test_split_on_pauses_cuts_in_silence():
    sr = stt_mod.SAMPLE_RATE
    x = np.full(50 * sr, 0.5, dtype=np.float32)
    x[15 * sr:int(15.1 * sr)] = 0                 # a pause at 15 s
    x[33 * sr:int(33.1 * sr)] = 0                 # and at 33 s
    segs = stt_mod.split_on_pauses(x)
    assert sum(map(len, segs)) == len(x)
    assert all(len(s) <= 22 * sr for s in segs)
    assert abs(len(segs[0]) / sr - 15.05) < 0.01
    assert stt_mod.split_on_pauses(x[:10 * sr])[0] is not None and len(stt_mod.split_on_pauses(x[:22 * sr])) == 1
