import os
import subprocess
from pathlib import Path

VIDEO_ROOT = "../audio-dataset/faceforensics"
AUDIO_ROOT = "../audio-dataset/faceforensics_audio"

os.makedirs(AUDIO_ROOT, exist_ok=True)

def extract(class_name):
    in_dir = os.path.join(VIDEO_ROOT, class_name)
    out_dir = os.path.join(AUDIO_ROOT, class_name)
    os.makedirs(out_dir, exist_ok=True)

    for vid in os.listdir(in_dir):
        if not vid.endswith(".mp4"):
            continue

        in_path = os.path.join(in_dir, vid)
        out_path = os.path.join(out_dir, vid.replace(".mp4", ".wav"))

        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            out_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for c in ["real", "fake"]:
    extract(c)

print("Audio extraction done.")