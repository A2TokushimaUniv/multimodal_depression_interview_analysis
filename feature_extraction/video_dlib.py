from utils import get_video_files, get_igaku_target, get_riko_target, save_feature
from logzero import logger
import os
import numpy as np
import cv2
import urllib.request
import bz2
import dlib
import pandas as pd


def _download_face_landmark_model():
    """
    DlibでのFace Landmark検出のためのモデルをダウンロードする
    """
    filename = "shape_predictor_68_face_landmarks.dat.bz2"
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    if not os.path.exists(filename):
        logger.info("Downloading face landmark model for dlib...")
        try:
            urllib.request.urlretrieve(url, filename)
            logger.info("Finished downloading face landmark model for dlib.")
        except Exception as e:
            logger.error(f"Failed to download face landmark model: {e}")
            return

    output_filename = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(output_filename):
        logger.info("Decompressing face landmark model for dlib...")
        try:
            with bz2.BZ2File(filename, "rb") as file:
                decompressed_data = file.read()
            with open(output_filename, "wb") as out_file:
                out_file.write(decompressed_data)
        except Exception as e:
            logger.error(f"Failed to decompress face landmark model: {e}")
            return
    return


def _get_dlib_feature(video_path, detector, predictor):
    """
    D-Vlogの元論文に記載されている方法でDlibの特徴量を取得する
    """
    cap = cv2.VideoCapture(video_path)

    # FPSを設定して1 FPSごとにフレームを処理
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 総フレーム数を取得
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 動画の長さ（秒）を計算
    video_duration = frame_count / fps

    frame_count = 0
    face_frame_count = 0
    noface_frame_count = 0
    landmark_vectors = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 1 FPSごとの処理を行う
        if frame_count % fps == 0:
            # フレームをグレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 顔を検出
            faces = detector(gray)
            if len(faces) == 0:
                # 顔が検出されない場合は0ベクトルで埋める
                landmarks_vector = np.zeros(136)
                noface_frame_count += 1
            else:
                # 最初に検出された顔に対してランドマークを取得
                landmarks = predictor(gray, faces[0])
                landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
                landmarks_vector = landmarks_array.flatten()
                face_frame_count += 1
            landmark_vectors.append(landmarks_vector)
        frame_count += 1
    cap.release()
    result = pd.DataFrame(landmark_vectors)
    if video_duration * 10 - result.shape[0] * 10 >= 10:
        logger.warning(
            f"duration: {video_duration} != dlib feature column: {result.shape[0]}"
        )
    return result


def extract_dlib_feature(input_data_dir):
    """
    Dlibの特徴量を抽出する
    """
    _download_face_landmark_model()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    riko_video_files, igaku_video_files = get_video_files(input_data_dir)
    for riko_video_file in riko_video_files:
        logger.info(f"Extracting Dlib feature from {riko_video_file}....")
        feature = _get_dlib_feature(riko_video_file, detector, predictor)
        data_id = riko_video_file.split("/")[-2]
        target = get_riko_target(data_id)
        save_feature(feature, os.path.join(input_data_dir, "dlib"), target)
    for igaku_video_file in igaku_video_files:
        logger.info(f"Extracting Dlib feature from {igaku_video_file}....")
        feature = _get_dlib_feature(igaku_video_file, detector, predictor)
        data_id = igaku_video_file.split("/")[-2]
        target = get_igaku_target(data_id)
        save_feature(feature, os.path.join(input_data_dir, "dlib"), target)
    return
