from .preprocess import (
    get_subject_frames,
    get_subject_timestamp,
    get_subject_audio,
    get_subject_text,
)
from .utils import make_dir

from pydub import AudioSegment
from logzero import logger


# 被験者の音声ファイルからテキストデータを抽出する
def _extract_text_feature(subject_audio_file):
    subject_text = get_subject_text(subject_audio_file)
    logger.info(f"被験者のテキスト: \n{subject_text}")
    # TODO: BERTなどで特徴抽出
    return


def _extract_sound_feature(subject_audio_file):
    # TODO: subject_audio_fileを読み取って、openSmileなどで特徴抽出
    # return sound_feature
    logger.info("音声特徴量")
    return


# カウンセリング動画データから特徴抽出を行う
def _extract_movie_feature(movie_file, subject_segments):
    # 音声データを元に被験者のみの動画データを抽出する
    subject_frame = get_subject_frames(movie_file, subject_segments)
    # TODO:フレームの確認
    # TODO: videomaeでの特徴抽出
    # return movie_feature
    logger.info("動画特徴量")
    return


# 動画データと音声データからから特徴抽出を行う
def extract_feature(movie_file, audio_file, output_dir):
    make_dir(output_dir)
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = get_subject_timestamp(audio, output_dir)
    # ↑を利用して音声データから被験者の音声データを抜き出す
    subject_audio_file = get_subject_audio(audio, subject_segments, output_dir)

    # 各モダリティの特徴抽出
    _extract_sound_feature(subject_audio_file, output_dir)
    _extract_movie_feature(movie_file, subject_segments, output_dir)
    _extract_text_feature(subject_audio_file, output_dir)
