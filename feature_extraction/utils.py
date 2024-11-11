import numpy as np
import os
import glob
import pandas as pd


def _save_as_npy(csv_file_path, output_dir):
    """
    CSVファイルを.npyファイル（numpy形式）に変換する
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.loadtxt(csv_file_path, delimiter=",", skiprows=1)
    npy_file_path = os.path.basename(os.path.splitext(csv_file_path)[0]) + ".npy"
    np.save(os.path.join(output_dir, npy_file_path), data)


def save_feature(feature: pd.DataFrame, output_dir, output_file_name):
    """
    特徴量を保存する
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, output_file_name)
    feature.to_csv(csv_file_path, index=False)
    _save_as_npy(csv_file_path, output_dir)


def get_voice_files(input_data_dir):
    """
    音声ファイルを取得する
    """
    pattern = os.path.join(input_data_dir, "*", "*.wav")
    file_paths = glob.glob(pattern, recursive=True)
    result = []
    for file_path in file_paths:
        data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
        result.append((data_id, file_path))
    result.sort()
    return result


def get_video_files(input_data_dir):
    """
    動画ファイルを取得する
    """
    pattern = os.path.join(input_data_dir, "*", "*.mp4")
    file_paths = glob.glob(pattern, recursive=True)
    result = []
    for file_path in file_paths:
        data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
        result.append((data_id, file_path))
    result.sort()
    return result


def get_text_files(input_data_dir):
    """
    テキストファイルを取得する
    """
    pattern = os.path.join(input_data_dir, "*", "*.csv")
    file_paths = glob.glob(pattern, recursive=True)
    result = []
    for file_path in file_paths:
        data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
        result.append((data_id, file_path))
    result.sort()
    return result


def get_openface_files(input_data_dir):
    pattern = os.path.join(input_data_dir, "*", "*.csv")
    file_paths = glob.glob(pattern, recursive=True)
    result = []
    for file_path in file_paths:
        data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
        result.append((data_id, file_path))
    result.sort()
    return result
