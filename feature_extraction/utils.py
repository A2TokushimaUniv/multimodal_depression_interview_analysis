import numpy as np
import os


def save_as_npy(csv_file, output_dir, skip_header=True):
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
