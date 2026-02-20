import os
import cv2
import torch

from model import load_model
from utils import DEVICE


model = load_model()
model.to(DEVICE)
model.eval()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
