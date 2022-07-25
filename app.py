import shutil
from fastapi import FastAPI, File, UploadFile, Depends, Body, Request
import uvicorn
from pydantic import BaseModel

from transformers.pipelines import pipeline
from transformers import AutoModelForCTC, AutoProcessor
import torch
import librosa
import soundfile as sf


app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)

processor = AutoTokenizer.from_pretrained("./wav2vec2_fine_tuned_fr")
model = AutoModelForSeq2SeqLM.from_pretrained("./wav2vec2_fine_tuned_fr").to(torch_device)


def transcription(audio,processor,model):
  speech, _ = librosa.load(audio, sr=16000, mono=True)
  inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
  with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
  pred = processor.batch_decode(logits.numpy()).text[0]
  return {'transcription':pred}

class Transcription:

    def __init__(self, audio_path : str):
        self.audio_path = audio_path
        self.speech_rate = 16000

    def transcription(self, long_model, long_processor):
        speech, _ = librosa.load(self.audio_path, sr=self.speech_rate, mono=True)
        inputs = long_processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = long_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        pred = long_processor.batch_decode(logits.numpy()).text[0]    
        return {'transcription' : pred} 

    
@app.get('/')
async def home():
    return {"message": "Hello World"}

@app.post("/transcription")
async def get_transcription(file: UploadFile):
    audio_path = f"app/media/{file.filename}"
    with open(audio_path, "wb+") as audio:
        shutil.copyfileobj(file.file, audio)
    STT = Transcription(audio_path)
    pred = STT.transcription(model, processor)
    return pred
