import librosa
import numpy as np
from pydub import AudioSegment
import simpleaudio as sa

def extract_features(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = np.mean(np.abs(y))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    return tempo, energy, contrast

def classify_vibe(tempo, energy, contrast):
    if tempo < 80 and energy < 0.02 and contrast < 20:
        return "Slow"
    elif 70 <= tempo <= 100 and energy < 0.03 and contrast < 25:
        return "Soothing"
    elif tempo > 120 and energy > 0.04 and contrast > 30:
        return "Party"
    elif 90 <= tempo <= 160 and 0.1 <= energy <= 0.14:
        return "Mood-setting"
    else:
        return "Unclassified"

def analyze_segments(file_path, segment_duration=10_000):  # 10 seconds in ms
    audio = AudioSegment.from_file(file_path)
    segments = []
    for i in range(0, len(audio), segment_duration):
        segment = audio[i:i + segment_duration]
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        if segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)  # convert to mono
        samples /= np.max(np.abs(samples))  # normalize
        y = samples
        sr = segment.frame_rate
        features = extract_features(y, sr)
        segments.append((i, i + segment_duration, features))
    return segments

def find_best_segment(segments, target_vibe):
    best_score = float('-inf')
    best_segment = None

    for start, end, (tempo, energy, contrast) in segments:
        # Define scores based on closeness to ideal vibe ranges
        score = 0
        if target_vibe == "Slow":
            score -= abs(tempo - 70)
            score -= abs(energy - 0.015) * 100
            score -= abs(contrast - 18)
        elif target_vibe == "Soothing":
            score -= abs(tempo - 85)
            score -= abs(energy - 0.02) * 100
            score -= abs(contrast - 22)
        elif target_vibe == "Party":
            score -= abs(tempo - 130)
            score -= abs(energy - 0.05) * 100
            score -= abs(contrast - 35)
        elif target_vibe == "Mood-setting":
            score -= abs(tempo - 100)
            score -= abs(energy - 0.03) * 100
            score -= abs(contrast - 28)

        if score > best_score:
            best_score = score
            best_segment = (start, end)

    return best_segment

def play_segment(file_path, start_ms, end_ms):
    audio = AudioSegment.from_file(file_path)[start_ms:end_ms]
    playback = sa.play_buffer(
        audio.raw_data,
        num_channels=audio.channels,
        bytes_per_sample=audio.sample_width,
        sample_rate=audio.frame_rate
    )
    playback.wait_done()

# --- Usage ---
file_path = "husn.mp3"

# Step 1: Analyze whole audio
y, sr = librosa.load(file_path, duration=60)
tempo, energy, contrast = extract_features(y, sr)
print(f"Tempo: {tempo}, Energy: {energy}, Contrast: {contrast}")
vibe = classify_vibe(tempo, energy, contrast)
print(f"Detected Vibe: {vibe}")

# Step 2: Find best segment for that vibe
segments = analyze_segments(file_path)
best = find_best_segment(segments, vibe)

# Step 3: Play it
if best:
    print(f"Playing best {vibe} segment: {best[0] // 1000}–{best[1] // 1000} seconds")
    play_segment(file_path, best[0], best[1])
else:
    print("No segment found!")
