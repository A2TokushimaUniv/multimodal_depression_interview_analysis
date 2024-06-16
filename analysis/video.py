import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import os
import glob
from logzero import logger


def calculate_vector(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vectors = []
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Analyze the frame using DeepFace
            try:
                vector = DeepFace.represent(
                    frame, model_name="OpenFace", enforce_detection=False
                )
                vectors.append(vector[0]["embedding"])
            except Exception as e:
                print(f"Error analyzing frame {frame_count}: {e}")
        frame_count += 1
    cap.release()

    change_rates = []
    for i in range(1, len(vectors)):
        prev_vector = vectors[i - 1]
        current_vector = vectors[i]
        cos_dist = cosine(prev_vector, current_vector)
        change_rate = abs(1 - cos_dist)
        change_rates.append(change_rate)
    return sum(change_rates) / len(change_rates)


def analyze_deepface(before_sum_df):
    riko_video_files = glob.glob(
        os.path.join("../data/preprocessed_data/video/riko", "**", "*-video*.mp4"),
        recursive=True,
    )
    igaku_video_files = glob.glob(
        os.path.join(
            "../data/preprocessed_data/video/igaku", "**", "*_zoom_映像・音声*.mp4"
        ),
        recursive=True,
    )

    for riko_video_file in riko_video_files:
        logger.info(f"Extracting DeepFace vector from {riko_video_file}.")
        average_change_rate = calculate_vector(riko_video_file)
        id = riko_video_file.split("/")[-2]
        if int(id) < 10:
            before_sum_df.loc[
                before_sum_df["タイムスタンプ"] == f"riko0{id}", "Average_change_rate"
            ] = average_change_rate
        else:
            before_sum_df.loc[
                before_sum_df["タイムスタンプ"] == f"riko{id}", "Average_change_rate"
            ] = average_change_rate

    for igaku_video_file in igaku_video_files:
        logger.info(f"Extracting DeepFace vector from {igaku_video_file}.")
        average_change_rate = calculate_vector(igaku_video_file)
        id = igaku_video_file.split("/")[-2]
        before_sum_df.loc[
            before_sum_df["タイムスタンプ"] == f"psy_c_{id}", "Average_change_rate"
        ] = average_change_rate

    return before_sum_df
