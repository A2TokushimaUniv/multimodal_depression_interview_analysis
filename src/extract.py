from preprocess import (
    get_subject_frames,
    get_subject_timestamp,
    get_subject_audio,
    get_subject_text,
)
from pydub import AudioSegment


# 被験者の音声ファイルからテキストデータを抽出する
def _extract_text_feature(subject_audio):
    subject_text = get_subject_text(subject_audio)
    # TODO: BERTなどで特徴抽出
    return


def _extract_sound_feature(subject_audio):
    # TODO: openSmileなどで特徴抽出
    return sound_feature


# カウンセリング動画データから特徴抽出を行う
def _extract_movie_feature(movie_file, sound_file):
    # 音声データを元に被験者のみの動画データを抽出する
    subject_frame = get_subject_frames(movie_file, sound_file)
    # TODO: videomaeでの特徴抽出
    return movie_feature


def output(movie_feature, sound_feature, text_feature, output_dir):
    # TODO: ファイルに出力、あまり形式が分かっていない
    return


# 動画データと音声データからから特徴抽出を行う
def extract_feature(movie_file, sound_file, output_dir):
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(sound_file)
    # 音声データから被験者の音声データを抜き出す
    subject_segments = get_subject_timestamp(audio)
    subject_audio = get_subject_audio(audio, subject_segments)

    # 各モダリティの特徴抽出
    sound_feature = _extract_sound_feature(subject_audio)
    movie_feature = _extract_movie_feature(movie_file, subject_segments)
    text_feature = _extract_text_feature(subject_audio)
    output(movie_feature, sound_feature, text_feature, output_dir)
