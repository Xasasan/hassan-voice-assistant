
# hugging face cache ni d diskga o'tkazdm .

import os
os.environ["HF_HOME"] = "D:/.cache/huggingface" 
os.environ["TRANSFORMERS_CACHE"] = "D:/.cache/huggingface/hub"
os.environ["TORCH_HOME"] = "D:/.cache/torch"  


import torch
print(torch.hub.get_dir())  


#____________________________________________________________

# MAN WAKE WORDS GA O'ZIMMI CUSTOM WORD DAN FOYDALANMOQCHIMAN VA OPENWAKEWORD DAN FOYDALANAMAN 
#_______________________________________________________

# Mic input (real time)
#        ↓
# Frame splitting (e.g., 80 ms chunks)
#        ↓
# Feature extraction
#        ↓
# Wake-word classifier
#        ↓
# Probability score
#        ↓
# Trigger when score > threshold

# #
#___________________________________________________________________
# Option 1 — Use openWakeWord Training Tools

# They have notebooks to generate synthetic data and train new models. You won’t necessarily need to record 200+ samples yourself.
# (Radio-quality best results still require good data.)

# ✅ openWakeWord = ready real-time wake-word engine



# ________________HUGGING FACE TOKEN in vs code__________________________-
# huggingface-cli login 


#____________________Wake word detection__________________________

# dataset  Speech Commands for wake word detection
from openwakeword.model import Model
import sounddevice as sd
import numpy as np
import torchaudio

# 2️⃣ How to check if your WAV file is 16kHz and mono

waveform, sample_rate = torchaudio.load("D:/uzinfocom/hugging_face tasks/task/own_voice/own_voice.wav")

print("Sample rate:", sample_rate)
print("Shape:", waveform.shape)

#_____________________________OPENWAKEWORD MODEL __________________________________

# blocksize means - How many audio samples we record at one time.
# Since your sampling rate is:
# 16000 samples per second
# Your block:
# 1280 samples
# means the model listens to about:
# 1280 / 16000 = 0.08 seconds
# So your program checks the microphone every ~80 milliseconds.
# That is why it feels real-time.

#_____________________________________________________________________

# audio.flatten() means . 

# to convert:

# [[0.01],
#  [0.02],
#  [0.05]]

# into:

# [0.01, 0.02, 0.05]

# Basically:

# 2D array  →  1D array

#___________________NEEDED INSTALLS__________________________________

# pip install datasets==2.14.6 soundsfile librosa
# 
# python -m pip install openwakeword
# 
# How to capture microphone audio in VS Code
# pip install sounddevice numpy 
# pip install tflite-runtime
# pip install silero-vad
# python -m pip install pyaudio
# pip install huggingface_hub==0.23.0 transformers==4.40.0
# huggingface-cli login da . Make calls to serverless Inference Providers ga ✅ qo'yish kere
# pip install sentencepiece 
# pip install huggingface_hub==0.30.0 transformers==4.44.0 datasets==2.20.0 tokenizers==0.19.1
# ! pip install git+https://github.com/speechbrain/speechbrain.git@develop
# pip install speechbrain==0.5.14 torchaudio==2.0.2
# pip install numpy==1.26.4
# pip install ipython
#____________________________________________________________________________

from transformers import pipeline
import torch
import openwakeword
import pyaudio
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# my custom wake word model instead of MIT model
oww_model = openwakeword.Model(
    wakeword_models=["D:/uzinfocom/hugging_face tasks/task/models/haaasaaan.tflite"]
)

def launch_fn(prob_threshold=0.7, debug=False):
    print("Listening for wake word 'Hassan'...")
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16_000,
        input=True,
        frames_per_buffer=1280
    )
    
    while True:
        audio_chunk = stream.read(1280) # 80 milisekund
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        prediction = oww_model.predict(audio_data)
        score = prediction["haaasaaan"]
        
        if debug:
            print({"haaasaaan": score})
        
        if score > prob_threshold:
            print("Wake word 'Hassan' detected!")
            stream.stop_stream()
            stream.close()
            pa.terminate()
            return True


## Simple picture

# audio_data explanation
# Microphone raw:   [-32768  ...  0  ...  +32767]
#                   ↓ divide by 32768
# Tensor format:    [ -1.0   ...  0  ...   +1.0 ]

# VAD understands:  -1.0 to 1.0 ✅
# VAD doesn't get:  -32768 to 32767 ❌
#______________________________________________________________________
#
transcriber = pipeline(
    "automatic-speech-recognition", model = "openai/whisper-base.en", device = device
)

#_____________________VAD ULASH _______________________________________
# Install first:
# pip install silero-vad

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# Load VAD model
vad_model = load_silero_vad()

def transcribe_with_vad():
    sampling_rate = transcriber.feature_extractor.sampling_rate
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, # audio quality format
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=512
    )
    
    print("Start speaking...")
    frames = []
    silence_count = 0
    
    while True:
        # 1. grab audio chunk from mic
        audio_chunk = stream.read(512)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        frames.append(audio_data)
        
        # 2. give chunk to VAD
        audio_tensor = torch.tensor(audio_data).float() / 32768.0
        speech_prob = vad_model(audio_tensor, 16000).item()    # returns 0(silence) nd 1(speech) 
        
        # 3. VAD says: is someone speaking?
        if speech_prob < 0.7:       # silence detected
            silence_count += 1
        else:
            silence_count = 0       # reset if speaking again
        
        # 4. if silent for 1 second → stop recording
        if silence_count > 50:      
            print("Speech ended!")
            break
    
    # 5. close mic
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    # 6. send full audio to Whisper
    full_audio = np.concatenate(frames)
    result = transcriber(full_audio.astype(np.float32) / 32768.0) 
    # meaning this number below vad bn whisper modeliga kickina sonlani beramz 32768.0 bo'lib, ula kickina sonlaga yaxshi chunadi kak normalizatsiya qilamz  
    return result["text"]

# So 32768 is the maximum possible value a microphone sample can have.
# Think of it like converting temperature:
# 
# Microphone numbers = Fahrenheit  (big numbers like -32768 to 32767)
# After dividing     = simple scale (-1.0 to 1.0)
# VAD and Whisper only understand the simple scale (-1.0 to 1.0), not the big microphone numbers.
# Just like converting dollars to a percentage — same value, different format that the AI expects!

 #___________________________________________________________________________

## The full flow with VAD

# 👂 "Hassan" detected
#        ↓
# 🎤 Mic opens
#        ↓
# You speak → VAD says "YES speech" → keep recording
# You speak → VAD says "YES speech" → keep recording
# You stop  → VAD says "NO speech"  → silence count +1
# Silence   → VAD says "NO speech"  → silence count +2
# ...12 times → STOP! ✅
#        ↓
# Send audio to Whisper
#        ↓
# 🌐 Query AI
#        ↓
# 🔊 Speak answer
# _______________________________________________________________________

# MIT model job — ONLY wake word detection
# Hears "marvin" → score 0.99 → WAKE UP! ✅
# Hears anything else → score low → keep waiting

# **That's it! Nothing else!**


# ## After MIT model — OTHER models take over

# MIT model hears "marvin"
#        ↓
# Whisper listens to your question    ← different model
#        ↓
# Falcon/Qwen generates answer        ← different model
#        ↓
# SpeechT5 speaks the answer          ← different model


#__________________________________________________________________________
# ## Simple comparison
# Course:
# MIT model → hears "marvin" + understands command
# (one model does everything)

# Your version:
# openWakeWord → hears "Hassan"        ✅ better
# VAD          → detects silence       ✅ smarter  
# Whisper      → understands anything  ✅ more powerful


# ______________________LLM model Qwen/Qwen2.5-72B-Instruct_____________________________________________

#______________Language model query   using Inference API________________________________________


from huggingface_hub import HfFolder
import requests

print("My token: ", HfFolder().get_token())

#_________________________________________________________________________

from huggingface_hub import InferenceClient
# imports the tool that talks to Hugging Face servers


def query(text, model_id="Qwen/Qwen2.5-72B-Instruct"):
    # text     = your question  e.g. "What is France?"
    # model_id = which AI to ask (given LLM MODEL Qwen 72B)
    client = InferenceClient(
        provider="novita",   # which company runs Qwen on their servers
        api_key=HfFolder().get_token()
    )

    # creates connection to Hugging Face cloud
    # like opening a phone call to Qwen

    print(f"Querying...: {text}")
    
    message = client.chat.completions.create(
        model=model_id, 
        messages=[{"role": "user", "content": text}],   # i am  the user,  my question goes to the context
        max_tokens=500  # answer max 500 words
    )
    
    return message.choices[0].message.content
# message.choices[0]  = first answer (there could be multiple)
# .message            = the message object (bu argument message  ni ichidigi)
# .content            = just the text of the answer (message ni ichida message argumenti bor va uni ichida content key bor oshani opchiqvomman )
# returns clean text answer ✅


# sends question to Qwen on novita servers
# waits for answer...
# saves full response in "message"

query("What does Hugging Face do?")



# ## Full flow visually
# ```
# You call query("What does Hugging Face do?")
#        ↓
# Opens connection to novita servers
#        ↓
# Sends: {"role": "user", "content": "What does Hugging Face do?"}
#        ↓
# Qwen 72B thinks on novita's computers...
#        ↓
# Returns: "Hugging Face is a company that provides AI tools..."
#        ↓
# message.choices[0].message.content extracts just the text
#        ↓
# Returns clean answer string ✅
# ```

# ---

# ## Simple analogy
# ```
# client           = your phone
# provider=novita  = phone network
# api_key          = your SIM card
# messages         = your text message
# Qwen 72B         = person you're texting
# .content         = their reply text

#___________________________________________________________________________
# Synthesise speech

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

#____________________bo'tta yana s2s projectimda qigan ishi qilaman vctk datasetidan o'zim audiodan speaker embedding chiqaraman using ECAPA MODEL__________________________________

# from datasets import load_dataset

# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

#________________________________________________________________________________________
# speechbrain ni install qilaman
# ! pip install git+https://github.com/speechbrain/speechbrain.git@develop

from datasets import load_dataset

dataset = load_dataset("sanchit-gandhi/vctk", split="train")
print("dataset: ",dataset)
#_____________________________
# so i am taking one  ['array'] inside of sample   from sanchit-gandhi/vctk dataset and give it to the ecapa and ecapa gives me speaker embedding what is next 
# i have speaker embedding now if i give it to the speecht5 model . my speecht5 model will talk only voice which is i gave it right 

sample = dataset[1000]['speaker_id']
print("sample: ", sample)

speaker_id = "p226"

speaker_samples = [sample for sample in dataset if sample["speaker_id"] == speaker_id]

# bo'tta bitta speaker ni hamma audiolarini chiqarib 1 ta listga qo'ydm

print("speaker_samples: ", speaker_samples[0:2])


import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb")

    
embeddings = []

for sample in speaker_samples:
    audio = sample["audio"]
    signal = audio["array"]
    fs = audio["sampling_rate"]

    waveform = torch.tensor(signal).unsqueeze(0).float()

    #  Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    #  Resample to 16kHz if needed
    if fs != 16_000:
        resampler = torchaudio.transforms.Resample(fs, 16_000)
        waveform = resampler(waveform)

    
    # Pass waveform to ECAPA
    emb = classifier.encode_batch(waveform)
    embeddings.append(emb.squeeze(0))
    
# You passed audio into ECAPA, and ECAPA generated the speaker embedding (192-dim vector) based on what it learned from VoxCeleb during training.
#  Average embeddings for a stable speaker embedding
speaker_embedding_192 = torch.mean(torch.stack(embeddings), dim = 0)

# Project to 512
speaker_embedding_512 = torch.nn.functional.pad(speaker_embedding_192, (0, 512-192))


print("Final embedding shape:", speaker_embedding_512.shape)  # torch.Size([1, 512])
#___________________________________
# This just zero-pads to 512.
# Will it work?
# Sometimes yes.
# Voice quality? Maybe unstable.

# But it avoids random projection.
#___________________________________________________________________

# Project to 512 tepada 192  dim dan 512 qilishim kere why cus ttst5 waits 512 dim speaker embeddings 
# bo'tta 1 ta speakerri hamma audiolarini 1 ta embedding ga qoyvomman nega 
# chunki ttst5 modelim 1 ta speakerri kop audiolaridan yaxshiro chunadi speakerri gapirishini yaxshiro o'rganadi .

#_______________________________________________________________________
# Final embedding shape: torch.Size([1, 1, 512]) shunaqa chiqudi 
# speaker_embedding_192 = torch.mean(torch.stack(embeddings), dim=0)   # ← remove .unsqueeze(0)
# unsqueeze(0)ni optashadm 

def synthesise(text):
    text = text[:500] 
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
    inputs["input_ids"].to(device), speaker_embedding_512.to(device), vocoder=vocoder)
    return speech.cpu()

#__________________________TO CHECK synthesise FUNCTION______________________


from IPython.display import Audio

audio = synthesise(
    "Hugging Face is a company that provides natural language processing and machine learning tools for developers."
)

Audio(audio, rate=16000)

#_________________________________________________________________________________________


# Now that we’ve defined a function for each of the four stages of the voice assistant pipeline
#  all that’s left to do is piece them together to get our end-to-end voice assistant. 
#  We’ll simply concatenate the four stages, starting with wake word detection (launch_fn), speech transcription, querying the LLM, and finally speech synthesis.

#________________________________ FULL PIPELINE CHECK ____________________________________________________


import sounddevice as sd
import numpy as np
import time



time.sleep(1.5)
launch_fn(debug = True)
print(transcribe_with_vad())
print(query(" what is UAE "))


def play_audio(audio):
    audio_np = audio.numpy()        
    sd.play(audio_np, samplerate=16000)
    sd.wait()                       # wait until finished 
    time.sleep(1.0)

play_audio(synthesise("hello everyone it is Ronaldo"))

#________________________________ FULL PIPELINE  ____________________________________________________

time.sleep(1.5)
launch_fn(debug=True)              #  waits for "Hassan"
transcription = transcribe_with_vad()       #  my question
response = query(transcription)    #  AI answers
audio = synthesise(response)       #  speaks answer
play_audio(audio)

#________________________________________________________________FULL PIPELINE____________________________

# PART 1: Wake Word Detection    → openWakeWord
# PART 2: Speech to Text         → Whisper + VAD
# PART 3: Generate Answer        → Qwen 72B (cloud)
# PART 4: Text to Speech         → SpeechT5 + ECAPA


# 👂 Always listening...
#        ↓
# You say "Hassan"
#        ↓ haaasaaan.tflite
# ✅ Wake word detected! (score > 0.7)
#        ↓
# 🎤 Mic opens "Start speaking..."
#        ↓
# You ask "What is the capital of France?"
#        ↓ silero VAD
# 🔇 Silence detected → stop recording
#        ↓ whisper-base.en
# 📝 "What is the capital of France?"
#        ↓ Qwen 72B (novita cloud)
# 💬 "The capital of France is Paris..."
#        ↓ ECAPA + SpeechT5 + HifiGan
# 🔊 Speaks answer in p226's voice
#        ↓
# 👂 Goes back to listening...

# QOGANI HOMEWORK GURUPPAMDA 
# voice asisstant Xasan project full pipeline YOZAMAN