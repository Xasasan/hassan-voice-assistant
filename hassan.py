"""
Hassan - custom voice assistant (refactored for a reliable live demo).

Pipeline:
  Say "Hassan"             -> openWakeWord (custom ONNX model) wakes it up
  -> Silero VAD + Whisper  -> listen until you stop talking, then transcribe
  -> Qwen 2.5 72B (HF Inference / Novita) -> generate a short answer
  -> SpeechT5 + HiFi-GAN   -> speak the answer out loud

Run (after one-time setup):  python hassan.py
One-time setup (needs internet):
    set HF_HUB_OFFLINE=0
    huggingface-cli login                # for the Qwen LLM
    python build_speaker_embedding.py    # creates speaker_embedding.pt (the voice)
    python hassan.py                     # first run downloads Whisper + SpeechT5
"""

import os
# Cache models on D: and load them from the LOCAL cache only (no network calls on startup).
# This avoids hangs on a flaky connection and keeps the live demo independent of room wifi.
# FIRST-TIME download on a fresh machine: run with HF_HUB_OFFLINE=0.
# Models cached on D: (HF_HUB_CACHE). The HF auth TOKEN stays at its DEFAULT location so a normal
# `huggingface-cli login` is picked up (overriding HF_HOME would make it look for the token on D:).
os.environ.setdefault("HF_HUB_CACHE", "D:/.cache/huggingface/hub")
os.environ.setdefault("TORCH_HOME", "D:/.cache/torch")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# NOTE: do NOT set HF_HUB_OFFLINE globally - it would also block the Qwen Inference API call.
# Each local model is loaded with local_files_only=True instead (cache-only, no network, no hang),
# while the Qwen call stays free to reach the internet.
# Avoid the OpenMP duplicate-runtime crash on Windows (onnxruntime + torch/numpy both load
# libiomp5md.dll, which otherwise causes a 0xC0000005 access violation).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
import torch
import pyaudio
import sounddevice as sd
from openwakeword.model import Model as WakeWordModel
from silero_vad import load_silero_vad
from transformers import (pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor,
                          SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan)
from huggingface_hub import HfFolder, InferenceClient

HERE = Path(__file__).resolve().parent
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[startup] device = {device}")

# ----------------------------- configuration --------------------------------
SAMPLE_RATE = 16_000                                   # everything runs at 16 kHz
WAKE_MODEL_PATH = HERE / "models" / "haaasaaan.onnx"   # FIX E2/E3: relative path + ONNX (no tflite-runtime needed)
WAKE_LABEL = "haaasaaan"                               # openWakeWord names the score by the model filename
WAKE_THRESHOLD = 0.7                                   # score above this = "Hassan" heard
SPEAKER_EMBEDDING_PATH = HERE / "speaker_embedding.pt" # FIX E4: precomputed once, just loaded here
WHISPER_MODEL = "openai/whisper-base.en"               # English speech-to-text
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"                # answer generator (cloud)
VAD_THRESHOLD = 0.5                                    # Silero speech probability cut-off
VAD_SILENCE_FRAMES = 40                                # ~1.3 s of quiet after speech -> stop recording

# --------------------- pre-flight checks (no silent failures) ---------------
# FIX E1/E2: fail loudly with a clear message instead of crashing deep inside a library.
if not WAKE_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Wake-word model not found: {WAKE_MODEL_PATH}")
if not SPEAKER_EMBEDDING_PATH.is_file():
    raise FileNotFoundError(
        f"Speaker embedding not found: {SPEAKER_EMBEDDING_PATH}\n"
        f"Build it once with:  python build_speaker_embedding.py"
    )
if not HfFolder.get_token():
    raise RuntimeError("No Hugging Face token found. Run:  huggingface-cli login")

# ------------------------------- load models --------------------------------
print("[startup] loading wake-word model (ONNX) ...")
oww_model = WakeWordModel(wakeword_models=[str(WAKE_MODEL_PATH)], inference_framework="onnx")

print(f"[startup] loading Whisper ({WHISPER_MODEL}) ...")
# local_files_only=True -> load from cache only (no network, no has_file HEAD-check timeout).
# Build the pipeline from an explicitly-loaded model + processor so it applies fully.
_w_model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL, local_files_only=True).to(device)
_w_proc = AutoProcessor.from_pretrained(WHISPER_MODEL, local_files_only=True)
transcriber = pipeline("automatic-speech-recognition", model=_w_model,
                       tokenizer=_w_proc.tokenizer, feature_extractor=_w_proc.feature_extractor,
                       device=device)

print("[startup] loading Silero VAD ...")
vad_model = load_silero_vad()

print("[startup] loading SpeechT5 TTS + HiFi-GAN ...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", local_files_only=True)
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", local_files_only=True).to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", local_files_only=True).to(device)
speaker_embedding = torch.load(SPEAKER_EMBEDDING_PATH).to(device)  # shape [1, 512]


# ------------------------------ pipeline stages -----------------------------
def wait_for_wake_word():
    """Block until the custom 'Hassan' wake word is detected."""
    pa = pyaudio.PyAudio()
    # openWakeWord wants 80 ms frames = 1280 samples at 16 kHz.
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                     input=True, frames_per_buffer=1280)
    print('\nListening for "Hassan" ...')
    try:
        while True:
            # exception_on_overflow=False: don't crash if the mic buffer briefly overflows.
            chunk = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
            score = oww_model.predict(chunk)[WAKE_LABEL]
            if score > WAKE_THRESHOLD:
                print('-> Wake word "Hassan" detected!')
                return
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()


def listen_and_transcribe():
    """Record from the mic until the speaker goes quiet (Silero VAD), then transcribe (Whisper)."""
    pa = pyaudio.PyAudio()
    # Silero VAD requires EXACTLY 512-sample frames at 16 kHz.
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                     input=True, frames_per_buffer=512)
    print("Speak now ...")
    frames, silence, heard_speech = [], 0, False
    try:
        while True:
            chunk = np.frombuffer(stream.read(512, exception_on_overflow=False), dtype=np.int16)
            frames.append(chunk)
            # Normalise int16 [-32768, 32767] -> float [-1, 1], which VAD/Whisper expect.
            prob = vad_model(torch.tensor(chunk, dtype=torch.float32) / 32768.0, SAMPLE_RATE).item()
            if prob >= VAD_THRESHOLD:
                heard_speech, silence = True, 0
            else:
                silence += 1
            # Only stop once we've actually heard speech AND then a stretch of quiet.
            if heard_speech and silence > VAD_SILENCE_FRAMES:
                break
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    return transcriber(audio)["text"].strip()


def query_llm(text):
    """Ask Qwen 2.5 72B (on Novita via HF Inference) for a short, speakable answer."""
    try:
        client = InferenceClient(provider="novita", api_key=HfFolder.get_token())
        reply = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are Hassan, a helpful voice assistant. "
                                              "Answer in 1-3 short sentences suitable for reading aloud."},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
        )
        return reply.choices[0].message.content.strip()
    except Exception as e:
        # No silent failure: report it and keep the demo alive with a spoken fallback.
        print(f"[error] LLM query failed: {e}")
        return "Sorry, I could not reach the language model right now."


def synthesise(text):
    """Turn text into speech with SpeechT5 + HiFi-GAN, using Hassan's voice embedding."""
    text = text[:550]  # SpeechT5 is happiest with short inputs; keeps answers snappy.
    inputs = processor(text=text, return_tensors="pt")
    speech = tts_model.generate_speech(inputs["input_ids"].to(device), speaker_embedding, vocoder=vocoder)
    return speech.cpu().numpy()  # 16 kHz float waveform


def play(audio):
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


# --------------------------------- main loop --------------------------------
def main():
    print('\nHassan is ready. Say "Hassan", then ask a question. Press Ctrl+C to quit.')
    while True:
        wait_for_wake_word()
        question = listen_and_transcribe()
        print(f"You: {question}")
        if not question:
            print("(heard nothing - going back to sleep)")
            continue
        answer = query_llm(question)
        print(f"Hassan: {answer}")
        play(synthesise(answer))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
