from scoring import compute_final_score

video = {"video_score": 0.8}
audio = {"audio_probability": 0.6}
meta  = {"metadata_score": 0.4, "recycled": False}

print(compute_final_score(video, audio, meta))