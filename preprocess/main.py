from pydub.silence import detect_nonsilent
import os
from pydub import AudioSegment
from logzero import logger
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips
import csv
from reazonspeech.nemo.asr import transcribe, audio_from_path
from reazonspeech.nemo.asr import load_model
from utils import (
    get_subject_voice_files,
    get_counsellor_voice_files,
    get_video_files,
    set_random_seed,
)

# ReazonSpeech model for Speech-to-Text
# See: https://huggingface.co/reazon-research/reazonspeech-nemo-v2
REAZON_MODEL = load_model(device="cuda")
IGNORE_SEGMENTS_MILLI_SECONDS = 1000


def _get_speech_segments(audio, min_silence_len=500, silence_thresh=-50):
    """
    音声データから発話区間の開始ミリ秒・終了ミリ秒を取得
    """
    logger.info("発話区間を取得しています...")
    # 音のある区間を抽出する
    nonsilent_segments = detect_nonsilent(
        audio,
        # min_silenceの大きさによって出力される発話の区間が変わる
        min_silence_len=min_silence_len,  # min_silence_len ミリ秒以上無音なら区間を抽出
        silence_thresh=silence_thresh,  # slice_thresh dBFS以下で無音とみなす
    )
    speech_segments = []
    for start, end in nonsilent_segments:
        if (
            end - start < IGNORE_SEGMENTS_MILLI_SECONDS
        ):  # IGNORE_SEGMENTS_MILLI_SECONDSミリ秒未満の区間は無視する
            logger.info(
                f"{start}ミリ秒から{end}ミリ秒の区間は{IGNORE_SEGMENTS_MILLI_SECONDS}ミリ秒未満であるため無視します"
            )
            continue
        speech_segments.append((start, end))
    logger.info("発話区間を取得しました")
    return speech_segments


def _get_subject_text_list(voice_file_path, start, end):
    """
    発話開始秒, 発話終了秒, 発話テキスト からなるCSV行を生成する
    """
    audio = audio_from_path(voice_file_path)
    try:
        logger.info(
            f"{voice_file_path}の{start}ミリ秒から{end}ミリ秒をテキスト化しています..."
        )
        result = transcribe(REAZON_MODEL, audio)
    except Exception as e:
        logger.error(f"テキスト化の際に例外が発生しました：{e}")
        return None
    text = result.text
    if len(text) > 0:
        return [start / 1000, end / 1000, text]  # ミリ秒を秒に直してからCSVに書き込む
    return None


def _get_voice_text(
    audio,
    speech_segments,
    voice_output_file_path,
    text_output_file_path,
    is_counsellor=False,
):
    """
    発話区間だけ音声データを抜き出す
    """
    subject_voice_sum = AudioSegment.empty()
    subject_text_list = []
    logger.info("音声とテキストを抽出しています...")
    utterance_count = 1
    for start, end in speech_segments:
        subject_voice_sum += audio[start:end]
        subject_voice = AudioSegment.empty()
        subject_voice += audio[start:end]
        tmp_utterance_path = f"./tmp_utterance_{utterance_count}.wav"
        # 発話ごとに音声を一時的に保存
        subject_voice.export(tmp_utterance_path, format="wav")
        # 発話ごとのテキストを抽出
        text_list = _get_subject_text_list(tmp_utterance_path, start, end)
        if text_list:
            subject_text_list.append(text_list)
        utterance_count += 1
        os.remove(tmp_utterance_path)
    # 発話区間のみの音声データを保存
    if not is_counsellor:
        subject_voice_sum.export(voice_output_file_path, format="wav")
        logger.info(f"{voice_output_file_path}に前処理済みの音声を保存しました")
    # 発話テキストを保存
    with open(text_output_file_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_seconds", "end_seconds", "text"])
        writer.writerows(subject_text_list)
    logger.info(f"{text_output_file_path}にテキストを保存しました")
    return


def _get_video(video_file, speech_segments, video_output_file_path):
    """
    speech_segmentsを使って、対象者の映っている区間の動画フレームを抜き出す
    """
    video = VideoFileClip(video_file)
    # 動画の切り抜きと保存
    clips = []
    for i, (start_ms, end_ms) in enumerate(speech_segments):
        start_time = start_ms / 1000  # 秒に変換
        end_time = end_ms / 1000  # 秒に変換
        clip = video.subclip(start_time, end_time)
        clips.append(clip)
    # 切り抜いた動画を結合
    final_clip = concatenate_videoclips(clips)
    # 10FPSで動画を保存する
    final_clip.write_videofile(video_output_file_path, codec="libx264", fps=10)

    # リソースを解放
    video.reader.close()
    video.audio.reader.close_proc()
    for clip in clips:
        clip.reader.close()
        clip.audio.reader.close_proc()
    logger.info(f"{video_output_file_path}に前処理済みの動画を保存しました")
    return


def _preprocess(
    video_file_path,
    voice_file_path,
    save_dir,
    text_output_file_path,
    is_counsellor=False,
):
    """
    前処理を行う
    """
    logger.info(f"{video_file_path}と{voice_file_path}の前処理を開始します...")
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(voice_file_path)
    # 音声データから対象者が喋っている区間のミリ秒を取得する
    speech_segments = _get_speech_segments(audio)

    voice_output_file_name = os.path.splitext(os.path.basename(voice_file_path))[0]
    voice_output_file_path = os.path.join(save_dir, f"{voice_output_file_name}.wav")
    # speech_segmentsを利用して音声データから発話区間の音声と発話テキストを抜き出す
    _get_voice_text(
        audio,
        speech_segments,
        voice_output_file_path,
        text_output_file_path,
        is_counsellor,
    )

    if not is_counsellor:
        video_output_file_name = os.path.splitext(os.path.basename(video_file_path))[0]
        video_output_file_path = os.path.join(save_dir, f"{video_output_file_name}.mp4")
        # speech_segmentsを利用して動画データから対象者の映っている動画フレームを抜き出す
        _get_video(video_file_path, speech_segments, video_output_file_path)
    return


def main(input_data_dir, output_data_dir):
    os.makedirs(output_data_dir, exist_ok=True)

    subject_voice_files = get_subject_voice_files(input_data_dir)
    counsellor_voice_files = get_counsellor_voice_files(input_data_dir)
    video_files = get_video_files(input_data_dir)

    if not (
        len(subject_voice_files) == len(counsellor_voice_files) == len(video_files)
    ):
        logger.error(
            f"被験者の音声ファイルの数{len(subject_voice_files)}、カウンセラーの音声ファイルの数{len(counsellor_voice_files)}、動画ファイルの数{len(video_files)}が一致しません"
        )
        raise ValueError(
            "被験者の音声ファイルの数、カウンセラーの音声ファイルの数、動画ファイルの数が一致しません"
        )
    logger.info(f"{len(subject_voice_files)}個のデータを前処理します")

    # 被験者データの前処理
    for voice_file, video_file in zip(subject_voice_files, video_files):
        voice_data_id = voice_file[0]
        voice_file_path = voice_file[1]
        video_data_id = video_file[0]
        video_file_path = video_file[1]

        if voice_data_id != video_data_id:
            logger.error(
                f"voice_data_id: {voice_data_id} != video_data_id: {video_data_id}"
            )
            raise ValueError("voice_data_idとvideo_data_idが一致しません")

        data_id = voice_data_id
        multimodal_save_dir = os.path.join(output_data_dir, data_id)
        os.makedirs(multimodal_save_dir, exist_ok=True)
        video_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        os.makedirs(os.path.join(output_data_dir, "subject_text"), exist_ok=True)
        _preprocess(
            video_file_path,
            voice_file_path,
            multimodal_save_dir,
            os.path.join(
                output_data_dir, "subject_text", f"{data_id}_{video_filename}.csv"
            ),
        )

    for voice_file, video_file in zip(counsellor_voice_files, video_files):
        voice_data_id = voice_file[0]
        voice_file_path = voice_file[1]
        video_data_id = video_file[0]
        video_file_path = video_file[1]

        if voice_data_id != video_data_id:
            logger.error(
                f"voice_data_id: {voice_data_id} != video_data_id: {video_data_id}"
            )
            raise ValueError("voice_data_idとvideo_data_idが一致しません")

        data_id = voice_data_id
        multimodal_save_dir = os.path.join(output_data_dir, data_id)
        os.makedirs(multimodal_save_dir, exist_ok=True)
        video_filename = os.path.splitext(os.path.basename(video_file_path))[0]
        os.makedirs(os.path.join(output_data_dir, "counsellor_text"), exist_ok=True)
        _preprocess(
            video_file_path,
            voice_file_path,
            multimodal_save_dir,
            os.path.join(
                output_data_dir, "counsellor_text", f"{data_id}_{video_filename}.csv"
            ),
            True,
        )
    logger.info("前処理は正常に終了しました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_dir",
        help="生データを格納しているディレクトリ",
        type=str,
        default="../data/raw",
    )

    parser.add_argument(
        "--output_data_dir",
        default="../data/preprocessed",
        help="前処理結果を保存するディレクトリ",
    )

    args = parser.parse_args()
    input_data_dir = args.input_data_dir
    output_data_dir = args.output_data_dir

    logger.info(f"入力ディレクトリ：{input_data_dir}")
    logger.info(f"出力ディレクトリ：{output_data_dir}")
    set_random_seed()  # Speech2Textモデルが常に同じ結テキスト果を返すようにシード値を設定
    main(input_data_dir, output_data_dir)
