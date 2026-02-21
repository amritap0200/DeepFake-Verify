import torch
from model import CRNN   # or wherever your model class is

MODEL_PATH = "crnn_audio_fake.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()