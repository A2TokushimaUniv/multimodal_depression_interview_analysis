from .preprocess import (
    get_subject_segments,
    get_subject_audio,
    get_subject_text,
    get_subject_frames,
)
from .utils import make_dir
from .text import get_bert_feature
from .audio import get_opensmile_feature
from .video import get_videmae_feature

from pydub import AudioSegment
from logzero import logger
import os
import pickle


# 被験者の音声ファイルからテキストデータを抽出する
def _extract_text_feature(subject_audio_file_path, output_dir):
    subject_text = get_subject_text(subject_audio_file_path, output_dir)
    # TODO: BERTなどで特徴抽出
    text_feature = get_bert_feature(subject_text)
    with open(os.path.join(output_dir, "text_feature.pickle"), mode="wb") as f:
        pickle.dump(text_feature, f)
    logger.info("Successfully extracted text feature!")
    return


def _extract_audio_feature(subject_audio_file_path, output_dir):
    # TODO: subject_audio_fileを読み取って、openSmileなどで特徴抽出
    audio_feature = get_opensmile_feature(subject_audio_file_path)
    with open(os.path.join(output_dir, "audio_feature.pickle"), mode="wb") as f:
        pickle.dump(audio_feature, f)
    logger.info("Successfully extracted audio feature!")
    return


# カウンセリング動画データから特徴抽出を行う
def _extract_video_feature(video_file_path, subject_segments, output_dir):
    # 音声データを元に被験者のみの動画データを抽出する
    subject_video_file_path = get_subject_frames(
        video_file_path, subject_segments, output_dir
    )
    # TODO:フレームの確認
    # TODO: videomaeでの特徴抽出
    video_feature = get_videmae_feature(subject_video_file_path)
    logger.info("Successfully extracted video feature!")
    return


# 動画データと音声データからから特徴抽出を行う
def extract_feature(video_file_path, audio_file_path, output_dir):
    make_dir(output_dir)
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file_path)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = get_subject_segments(audio, output_dir)
    # ↑を利用して音声データから被験者の音声データを抜き出す
    subject_audio_file_path = get_subject_audio(audio, subject_segments, output_dir)

    # 各モダリティの特徴抽出
    # _extract_audio_feature(subject_audio_file_path, output_dir)
    _extract_video_feature(video_file_path, subject_segments, output_dir)
    # _extract_text_feature(subject_audio_file_path, output_dir)
