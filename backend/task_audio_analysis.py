from celery_app import celery_app
import librosa
import torch
import numpy as np
from model_loader import model, device

def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)
    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

@celery_app.task
def analyze_audio(file_path):
    mel = extract_mel(file_path).to(device)

    with torch.no_grad():
        output = model(mel)
        prob = torch.sigmoid(output).item()

    return {
        "fake_probability": prob,
        "label": "fake" if prob > 0.5 else "real"
    }