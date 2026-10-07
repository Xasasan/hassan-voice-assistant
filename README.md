# Hassan: custom wake-word voice assistant

Say **"Hassan"** and ask a question out loud. Hassan transcribes it, thinks, and answers in speech.
Built from separate open models wired into one real-time loop.

```
mic ─► openWakeWord (custom "Hassan" model) ─► Silero VAD ─► Whisper (base.en)
    ─► Qwen2.5-72B-Instruct (HF Inference API) ─► SpeechT5 + HiFi-GAN ─► speaker
```

| Stage | Model | Notes |
|---|---|---|
| Wake word | custom **openWakeWord** model trained on "Hassan" (`models/haaasaaan.onnx`, also `.tflite`) | threshold 0.7 |
| Voice activity | Silero VAD | stops recording after ~1.3 s of silence |
| Speech-to-text | `openai/whisper-base.en` | runs locally |
| Answer | `Qwen/Qwen2.5-72B-Instruct` via HF Inference | graceful fallback if offline |
| Text-to-speech | `microsoft/speecht5_tts` + HiFi-GAN vocoder | voice = x-vector speaker embedding |

## Files
- `hassan.py`: the assistant (refactored for a reliable live demo: relative paths, ONNX wake word, one clean loop, pre-flight checks, local models cache-only so a bad Wi-Fi connection can't hang it)
- `build_speaker_embedding.py`: builds the TTS voice from a CMU-Arctic x-vector
- `build_voice_from_recording.py`: alternative: builds the voice embedding from your own recording (speechbrain x-vector)
- `original/unit_7_voice_assistant.py`: the first version, kept for comparison

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
huggingface-cli login                 # for the Qwen Inference API
HF_HUB_OFFLINE=0 python build_speaker_embedding.py
python hassan.py
```
Model cache paths at the top of `hassan.py` point to `D:/.cache`; change them for your machine.

Built as the final project of the Hugging Face Audio Course, then extended and hardened for a live demo.
