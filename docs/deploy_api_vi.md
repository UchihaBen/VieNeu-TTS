# Chạy VieNeu-TTS API trên server GPU

Server API là `apps/openai_speech.py`, dùng chuẩn OpenAI `POST /v1/audio/speech`. Fork này thêm ba thứ:
- Trang nghe thử ở `/`.
- Giọng clone được lưu vào `voices/custom_voices.json`, nên khởi động lại server không bị mất.
- `DELETE /v1/voices/{name}` để xoá giọng đã clone.

## Cài đặt (làm 1 lần)

```bash
cd ~/VieNeu-TTS
uv sync --extra cuda
```

Model (`pnnbao-ump/VieNeu-TTS-v3-Turbo` và codec MOSS) tự tải vào `~/.cache/huggingface` khi chạy lần đầu. Không cần tải tay.
Model cần khoảng 1 GB VRAM. GPU từ đời Ampere trở lên (RTX 30xx/40xx, A-series) chạy bf16, tốt nhất.

## Chạy

```bash
tmux new -s tts        # hoặc: tmux attach -t tts
cd ~/VieNeu-TTS
HOST=0.0.0.0 PORT=8014 VIENEU_API_KEY="<key>" uv run python -m apps.openai_speech
```

Khi log hiện `✅ ready in ...s: backend=pytorch ...` và `Uvicorn running on http://0.0.0.0:8014` là server đã chạy.
Bấm `Ctrl+B` rồi `D` để thoát tmux mà server vẫn chạy.
Nếu log ghi `backend=onnx` nghĩa là server không nhận GPU mà đang chạy CPU. Kiểm tra lại bằng `nvidia-smi`.

- Trang nghe thử: `http://<ip-server>:8014/`. Dán key vào ô API key.
- Swagger: `http://<ip-server>:8014/docs`
- Kiểm tra trạng thái: `curl http://localhost:8014/health`

Một số biến môi trường hay dùng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Đặt `HOST=0.0.0.0` để máy khác gọi được |
| `VIENEU_API_KEY` | (trống = không khoá) | Client gửi `Authorization: Bearer <key>` |
| `VIENEU_MAX_STREAMS` | `16` | Số request chạy song song trên GPU |
| `VIENEU_VOICES_FILE` | `voices/custom_voices.json` | Nơi lưu giọng clone |
| `VIENEU_BACKBONE` | (model gốc) | Thư mục model đã fine-tune, ví dụ `finetune/output/ban_mai_v1/merged` (chỉ chạy GPU) |
| `VIENEU_EDU_NORMALIZE` | `1` | Chuẩn hoá chữ cho lớp học: `12 : 4` → chia, `1/2` → một phần hai, `3,14` → ba phẩy mười bốn, số La Mã. Tắt từng request bằng `"edu_normalize": false` |
| `VIENEU_WATERMARK` | `1` | `0` để tắt watermark âm thanh |

## Gọi API

```bash
KEY="<key>"; URL=http://<ip-server>:8014

# Danh sách giọng
curl -H "Authorization: Bearer $KEY" $URL/v1/voices

# Tạo giọng nói (giọng có sẵn)
curl -X POST $URL/v1/audio/speech -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là VieNeu TTS.", "voice": "Mai Anh", "response_format": "wav"}' -o out.wav

# Clone giọng: upload 3–8 giây audio sạch, sau đó dùng "voice": "<tên>"
curl -X POST $URL/v1/voices -H "Authorization: Bearer $KEY" -F name="Giong Cua Toi" -F file=@my_voice.wav

# Xoá giọng đã clone
curl -X DELETE "$URL/v1/voices/Giong%20Cua%20Toi" -H "Authorization: Bearer $KEY"
```

Python (dùng thư viện `openai`):
```python
from openai import OpenAI
client = OpenAI(base_url="http://<ip-server>:8014/v1", api_key="<key>")
with client.audio.speech.with_streaming_response.create(
    model="vieneu-v3-turbo", voice="Mai Anh", input="Xin chào!", response_format="wav",
) as r:
    r.stream_to_file("out.wav")
```

Các tham số khác của `/v1/audio/speech`:
- `sample_rate`: 48000, 24000, 16000 hoặc 8000.
- `response_format`: `wav` hoặc `pcm`.
- `stream_format`: `audio` hoặc `sse`.
- Tham số sinh: `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_chars`.

Xem chi tiết ở `docs/streaming.vi.md`.

Lưu ý: file WAV trả về là dạng stream, nên header ghi độ dài "không xác định". Hầu hết trình phát vẫn mở được. Nếu một công cụ nào đó báo lỗi, hãy dùng `response_format: "pcm"` (s16le mono, tần số lấy từ header `X-Sample-Rate`) rồi tự ghép header WAV.
