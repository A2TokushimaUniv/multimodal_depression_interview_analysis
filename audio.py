import librosa
import numpy as np

# 読み込む音声ファイル
audio_file = (
    "./data/preprocessed_data/voice/riko/1/audioNLPTarouTokushim11736812730.mp3"
)

# 音声ファイルを読み込む
y, sr = librosa.load(audio_file, sr=None)

# tempo, _ = librosa.beat.beat_track(y)
# print(tempo)

# 基本周波数 (F0) を計算する
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
)
f0 = f0[~np.isnan(f0)]

# 基本周波数の平均値を計算
pitch_mean = f0.mean()

print("Average Pitch:", pitch_mean)
