import glob
import os


def get_voice_files(input_data_dir):
    patterns = [
        os.path.join(
            input_data_dir, "*", "audioNLPTarou*.m4a"
        ),  # 理工学部データの音声データ
        os.path.join(
            input_data_dir, "*", "*_zoom_音声_被験者*.m4a"
        ),  # 医学部データの音声データ
    ]

    result = []
    for pattern in patterns:
        file_paths = glob.glob(pattern, recursive=True)
        for file_path in file_paths:
            data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
            result.append((data_id, file_path))
    result.sort()
    return result


def get_video_files(input_data_dir):
    pattern = os.path.join(input_data_dir, "*", "*.mp4")
    file_paths = glob.glob(pattern, recursive=True)
    result = []
    for file_path in file_paths:
        data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
        result.append((data_id, file_path))
    result.sort()
    return result
