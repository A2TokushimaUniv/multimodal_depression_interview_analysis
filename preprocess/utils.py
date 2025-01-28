import glob
import os

# NOTE: ファイルパターンが変われば追加する
subject_voice_file_patterns = [
    "*_zoom_音声_被験者.m4a",
    "audio和幸松本*.m4a",
    "audioNLPTarouTokushim*.m4a",
    "被験者.m4a",
]

counsellor_voice_file_patterns = [
    "*_zoom_音声_アバター.m4a",
    "*_zoom_音声_アバター.m4a",
    "audioハル*.m4a",
]


def get_subject_voice_files(input_data_dir):
    result = []
    for pattern in subject_voice_file_patterns:
        full_pattern = os.path.join(input_data_dir, "*", pattern)
        file_paths = glob.glob(full_pattern, recursive=True)
        for file_path in file_paths:
            data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
            result.append((data_id, file_path))
    result.sort()
    return result


def get_counsellor_voice_files(input_data_dir):
    result = []
    for pattern in counsellor_voice_file_patterns:
        full_pattern = os.path.join(input_data_dir, "*", pattern)
        file_paths = glob.glob(full_pattern, recursive=True)
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
