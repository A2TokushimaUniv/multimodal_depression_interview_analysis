import numpy as np
import opensmile


# 読み込む音声ファイル
audio_file = "./test.wav"

# # librosaでのピッチ計算
# y, sr = librosa.load(audio_file, sr=None)
# f0, voiced_flag, voiced_probs = librosa.pyin(
#     y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
# )
# f0 = f0[~np.isnan(f0)]
# pitch_mean = f0.mean()
# print("Average Pitch:", pitch_mean)

# # Librosaでのスピーチレートの計算
# frames = librosa.util.frame(y, frame_length=512, hop_length=64)
# speech_rate = np.sum(frames != 0) / (len(y) / sr)
# print(f"Speech rate: {speech_rate:.2f} words per second")


# OpenSMILEでのピッチ計算
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
features = smile.process_file(audio_file)
pitch_values = features["F0semitoneFrom27.5Hz_sma3nz_amean"]
print("Average Pitch:", pitch_values.mean())


# OpenSMILEでのジッター計算
loudness = features["loudness_sma3_amean"]
print(loudness)
mean_loudness = np.mean(loudness)
print(f"Mean Loudness: {mean_loudness}")

jitters = features["jitterLocal_sma3nz_amean"]
mean_jitter = np.mean(jitters)
print(f"Mean Jitter: {mean_jitter}")
