from .preprocess import (
    get_subject_frames,
    get_subject_timestamp,
    get_subject_audio,
    get_subject_text,
)
from .utils import remove_tmp_file

from pydub import AudioSegment
from logzero import logger
import os


# 被験者の音声ファイルからテキストデータを抽出する
def _extract_text_feature(subject_audio_file):
    subject_text = get_subject_text(subject_audio_file)
    logger.info(subject_text)
    # TODO: BERTなどで特徴抽出
    return


def _extract_sound_feature(subject_audio_file):
    # TODO: openSmileなどで特徴抽出
    # return sound_feature
    logger.info("音声特徴量")
    return


# カウンセリング動画データから特徴抽出を行う
def _extract_movie_feature(movie_file, subject_segments):
    # 音声データを元に被験者のみの動画データを抽出する
    subject_frame = get_subject_frames(movie_file, subject_segments)
    # TODO: videomaeでの特徴抽出
    # return movie_feature
    logger.info("動画特徴量")
    return


# def output(movie_feature, sound_feature, text_feature, output_dir):
#     # TODO: ファイルに出力、あまり形式が分かっていない
#     return


# 動画データと音声データからから特徴抽出を行う
def extract_feature(movie_file, audio_file, output_dir):
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file)
    # 音声データから被験者の音声データを抜き出す
    subject_segments = get_subject_timestamp(audio)
    subject_audio = get_subject_audio(audio, subject_segments)

    # TODO: TemporaryDirectoryで書き換える
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    TMP_AUDIO_FILE_PATH = (
        "tmp/subject_" + os.path.splitext(os.path.basename(audio_file))[0] + ".mp3"
    )
    logger.info(TMP_AUDIO_FILE_PATH)
    subject_audio.export(TMP_AUDIO_FILE_PATH, format="mp3")

    # 各モダリティの特徴抽出
    sound_feature = _extract_sound_feature(TMP_AUDIO_FILE_PATH)
    movie_feature = _extract_movie_feature(movie_file, subject_segments)
    text_feature = _extract_text_feature(TMP_AUDIO_FILE_PATH)
    # output(movie_feature, sound_feature, text_feature, output_dir)
    remove_tmp_file(TMP_AUDIO_FILE_PATH)
