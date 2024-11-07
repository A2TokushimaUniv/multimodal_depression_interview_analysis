import numpy as np
import os
import glob
import pandas as pd


def save_as_npy(csv_file, output_dir):
    """
    CSVファイルを.npyファイル（numpy形式）に変換する
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    npy_file_path = os.path.basename(os.path.splitext(csv_file)[0]) + ".npy"
    np.save(os.path.join(output_dir, npy_file_path), data)


def save_feature(feature: pd.DataFrame, output_dir, target):
    """
    特徴量を保存する
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, f"{target}.csv")
    feature.to_csv(csv_file_path, index=False)
    save_as_npy(csv_file_path, output_dir + "_npy")


def get_riko_target(data_id):
    if int(data_id) < 10:
        target = f"riko0{data_id}"
    else:
        target = f"riko{data_id}"
    return target


def get_igaku_target(data_id):
    target = f"psy_c_{data_id}"
    return target


def get_voice_files(input_data_dir):
    """
    音声ファイルを取得する
    """
    riko_voice_files = glob.glob(
        os.path.join(input_data_dir, "voice", "riko", "*", "audioNLP*.wav"),
        recursive=True,
    )
    igaku_voice_files = glob.glob(
        os.path.join(input_data_dir, "voice", "igaku", "*", "*_zoom_音声_被験者*.wav"),
        recursive=True,
    )
    return riko_voice_files, igaku_voice_files


def get_video_files(input_data_dir):
    """
    動画ファイルを取得する
    """
    riko_video_files = glob.glob(
        os.path.join(input_data_dir, "video", "riko", "*", "*.mp4"),
        recursive=True,
    )
    igaku_video_files = glob.glob(
        os.path.join(input_data_dir, "video", "igaku", "*", "*.mp4"),
        recursive=True,
    )
    return riko_video_files, igaku_video_files


def get_text_files(input_data_dir):
    """
    動画ファイルを取得する
    """
    riko_text_files = glob.glob(
        os.path.join(input_data_dir, "text", "riko", "*", "*.txt"),
        recursive=True,
    )
    igaku_text_files = glob.glob(
        os.path.join(input_data_dir, "text", "igaku", "*", "*.txt"),
        recursive=True,
    )
    return riko_text_files, igaku_text_files


def get_openface_files(input_data_dir):
    return glob.glob(os.path.join(input_data_dir, "openface", "*.csv"), recursive=True)
