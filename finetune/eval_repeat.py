"""
Measure how often a model repeats / adds words at the end of a reading.

Generates each test text N times with a model + voice, transcribes every take
with Whisper, and counts takes that say >= --extra-words more words than the
text (the "hay chủ động thay đổi hay chủ động thay đổi" failure). Compare the
base model and fine-tuned runs to pick a checkpoint that does not loop.

    uv run python finetune/eval_repeat.py --model finetune/output/ban_mai_v2/merged --voice "Ban Mai"
    uv run python finetune/eval_repeat.py --clip finetune/dataset/raw_audio/0001.mp3   # base model + clone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from check_dataset import normalize   # noqa: E402

TEXTS = [
    "Bạn biết không? Cuộc sống đôi khi rất kỳ lạ... Có những điều tưởng chừng đơn giản, nhưng lại khiến "
    "chúng ta suy nghĩ rất lâu! Vậy câu hỏi đặt ra là: Chúng ta nên tiếp tục chờ đợi, hay chủ động thay đổi?",
    "Vậy câu hỏi đặt ra là: Chúng ta nên tiếp tục chờ đợi, hay chủ động thay đổi?",
    "Các em hãy quan sát hình vẽ và cho cô biết có bao nhiêu con chim đang đậu trên cành cây.",
    "Mùa thu đến, những chiếc lá vàng rơi xào xạc trên sân trường. Các bạn học sinh tung tăng cắp sách đến lớp, "
    "gương mặt ai cũng rạng rỡ. Cô giáo đứng ở cửa lớp, mỉm cười chào đón từng em.",
    "Theo em, chúng ta nên đi bộ đến trường, hay đi xe đạp cho nhanh hơn?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="merged model folder (default: base model)")
    ap.add_argument("--voice", default=None, help="voice name known to the model")
    ap.add_argument("--clip", default=None, help="or clone this clip as the voice")
    ap.add_argument("--n", type=int, default=8, help="takes per text")
    ap.add_argument("--extra-words", type=int, default=3)
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--save-dir", default=None, help="also save every take as .wav here")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    if not args.voice and not args.clip:
        sys.exit("pass --voice or --clip")

    import librosa
    import numpy as np
    import torch
    from transformers import pipeline
    from vieneu import Vieneu

    kw = {"backbone_repo": args.model} if args.model else {}
    tts = Vieneu(mode="v3turbo", **kw)
    voice = args.voice
    if args.clip:
        voice = "__eval_clone__"
        tts.add_voice(voice, args.clip)
    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=args.asr,
                   torch_dtype=torch.float16 if cuda else torch.float32, device=0 if cuda else -1)
    save = Path(args.save_dir) if args.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    total_bad = total = 0
    for ti, text in enumerate(TEXTS, 1):
        ref = normalize(text)
        bad = 0
        for k in range(args.n):
            audio = np.asarray(tts.infer(text, voice=voice), dtype=np.float32)
            if save:
                tts.save(audio, str(save / f"t{ti}_{k}.wav"))
            x16 = librosa.resample(audio, orig_sr=tts.sample_rate, target_sr=16000)
            hyp = normalize(asr({"raw": x16, "sampling_rate": 16000},
                                generate_kwargs={"language": "vi", "task": "transcribe"})["text"])
            if len(hyp) - len(ref) >= args.extra_words:
                bad += 1
                print(f"  t{ti} take {k}: +{len(hyp) - len(ref)} words: ...{' '.join(hyp[-12:])}")
        total_bad += bad
        total += args.n
        print(f"text {ti}: {bad}/{args.n} takes with extra words")
    print(f"\nREPEAT RATE {total_bad}/{total} = {total_bad / total:.0%}   "
          f"(model={args.model or 'base'}, voice={args.voice or args.clip})")


if __name__ == "__main__":
    main()
