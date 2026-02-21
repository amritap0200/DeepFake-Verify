from task_audio_analysis import analyze_audio

# put any wav file path here
audio_path = "/home/hp/projects/spock--/audio-dataset2/real/LJ001-0001.wav"

result = analyze_audio.delay(audio_path)

print("Task sent! ID:", result.id)