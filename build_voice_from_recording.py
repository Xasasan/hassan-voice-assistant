"""
Build Hassan's voice from a recording (default: own_voice/own_voice.wav) -> speaker_embedding.pt.

Extracts a 512-dim speaker x-vector with speechbrain's spkrec-xvect-voxceleb - the SAME model
SpeechT5's built-in speaker embeddings were made with - so SpeechT5 speaks in the recorded
person's voice. (Your original used ECAPA-192 zero-padded to 512; this x-vector route is the
correct, stable 512-dim match for SpeechT5.)

Run once (needs internet for the one-time extractor download):
    python build_voice_from_recording.py
Use a different recording:
    python build_voice_from_recording.py  path\to\my_voice.wav
"""
import os
import sys
import shutil
os.environ.setdefault("HF_HOME", "D:/.cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# This builder needs the network the first time (to fetch the x-vector model):
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from huggingface_hub import snapshot_download
from speechbrain.pretrained import EncoderClassifier  # speechbrain 0.5.x import path

# speechbrain 0.5.x links model files with a Windows symlink, which needs admin / Developer Mode
# (WinError 1314). Patch symlink_to to fall back to a plain copy so it works without privileges.
_ORIG_SYMLINK = Path.symlink_to
def _symlink_or_copy(self, target, target_is_directory=False):
    try:
        return _ORIG_SYMLINK(self, target, target_is_directory)
    except OSError:
        src, dst = Path(target), self
        if src.resolve() == dst.resolve() or dst.exists():
            return  # file is already where it needs to be
        shutil.copy(str(src), str(dst))
Path.symlink_to = _symlink_or_copy

HERE = Path(__file__).resolve().parent
DEFAULT_WAV = HERE / "own_voice" / "own_voice.wav"
WAV = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WAV
OUT = HERE / "speaker_embedding.pt"
XVECT_DIR = HERE / "pretrained_models" / "spkrec-xvect-voxceleb"

if not WAV.is_file():
    raise FileNotFoundError(f"Recording not found: {WAV}")

print("Loading x-vector extractor (speechbrain/spkrec-xvect-voxceleb) ...")
# Pre-fetch the model files to a local folder with huggingface_hub (copies on Windows).
snapshot_download("speechbrain/spkrec-xvect-voxceleb", local_dir=str(XVECT_DIR))
classifier = EncoderClassifier.from_hparams(source=str(XVECT_DIR), savedir=str(XVECT_DIR))

print(f"Reading {WAV.name} ...")
# soundfile reads the WAV reliably on Windows; torchaudio handles the resample.
data, sr = sf.read(str(WAV), dtype="float32")
if data.ndim > 1:                                         # stereo -> mono
    data = data.mean(axis=1)
signal = torch.from_numpy(data).unsqueeze(0)              # [1, num_samples]
if sr != 16000:                                           # x-vector model expects 16 kHz
    signal = torchaudio.transforms.Resample(sr, 16000)(signal)

# encode_batch -> [1, 1, 512]; squeeze the batch dim -> [1, 512] (what SpeechT5 wants)
embedding = classifier.encode_batch(signal).squeeze(0)
# SpeechT5's reference (cmu-arctic) x-vectors are L2-normalised (norm 1.0); match that scale,
# otherwise the embedding is wildly out of range and the voice comes out muffled/unclear.
embedding = torch.nn.functional.normalize(embedding, dim=-1)
torch.save(embedding, OUT)
print(f"Saved voice embedding -> {OUT}   shape={tuple(embedding.shape)}  (expected (1, 512))")
