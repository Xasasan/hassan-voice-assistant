"""
Build Hassan's STANDARD (non-cloned) voice -> speaker_embedding.pt.

Uses a clean cmu-arctic x-vector - the 'slt' voice from the SpeechT5 documentation (clear, neutral).
hassan.py loads this at startup. This is the reliable, clear default voice.
(To instead clone a recording, use build_voice_from_recording.py.)

The x-vectors ship inside spkrec-xvect.zip in the dataset repo (load_dataset hits a UnicodeDecode
bug with this dataset), so we download + unzip directly.

Run once (needs internet for the one-time download):
    python build_speaker_embedding.py
"""
import os
os.environ.setdefault("HF_HUB_CACHE", "D:/.cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path
import zipfile
import glob
import numpy as np
import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
OUT = HERE / "speaker_embedding.pt"
EXTRACT_DIR = HERE / "pretrained_models" / "cmu-arctic-xvectors"

print("Downloading cmu-arctic x-vectors (one-time, small) ...")
zip_path = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(EXTRACT_DIR)

npys = sorted(glob.glob(str(EXTRACT_DIR) + "/**/*.npy", recursive=True))
# 'slt' = the clean US-female voice used in the SpeechT5 docs.
slt = [p for p in npys if "slt" in Path(p).name.lower()]
chosen = slt[0] if slt else npys[0]

embedding = torch.tensor(np.load(chosen), dtype=torch.float32)
if embedding.dim() == 1:
    embedding = embedding.unsqueeze(0)  # -> [1, 512]
torch.save(embedding, OUT)
print(f"Saved -> {OUT}   (voice: {Path(chosen).name}, shape={tuple(embedding.shape)})")
