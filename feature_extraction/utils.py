import numpy as np
import os
import glob


def _save_as_npy(csv_file, output_dir, skip_header=True):
    """
    CSVファイルを.npyファイル（numpy形式）に変換する
    """
    os.makedirs(output_dir, exist_ok=True)
    if skip_header:
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    else:
        data = np.loadtxt(csv_file, delimiter=",")
    npy_file_path = os.path.basename(os.path.splitext(csv_file)[0]) + ".npy"
    np.save(os.path.join(output_dir, npy_file_path), data)


def save_feature(feature, output_dir, target):
    """
    特徴量を保存する
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, f"{target}.csv")
    feature.to_csv(csv_file_path, index=False)
    _save_as_npy(csv_file_path, output_dir + "_npy")


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
