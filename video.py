import cv2
import numpy as np
import subprocess

# 動画ファイルのパス
video_path = "./data/preprocessed_data/video/igaku/1/001_zoom_映像・音声.mp4_10.mp4"
output_csv = "output.csv"

# OpenFaceのFeatureExtraction実行コマンド
cmd = [
    "./FeatureExtraction",  # OpenFaceのFeatureExtractionバイナリのパス
    "-f",
    video_path,
    "-of",
    output_csv,
]

# OpenFaceのFeatureExtractionを実行
subprocess.run(cmd)

# CSVファイルを読み込み
import pandas as pd

df = pd.read_csv(output_csv)

# 顔のランドマーク位置を抽出
landmarks = df.filter(regex="^x_|^y_")

# 表情変動率を計算
landmark_diff = landmarks.diff().dropna()
variation_rate = np.sqrt((landmark_diff**2).sum(axis=1))

# 結果をプロット
import matplotlib.pyplot as plt

plt.plot(variation_rate)
plt.title("Facial Expression Variation Rate")
plt.xlabel("Frame")
plt.ylabel("Variation Rate")
plt.show()
