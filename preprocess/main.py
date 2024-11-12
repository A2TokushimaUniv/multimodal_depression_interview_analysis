from pydub.silence import detect_nonsilent
import os
from pydub import AudioSegment
from logzero import logger
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips
import csv
from reazonspeech.nemo.asr import transcribe, audio_from_path
from reazonspeech.nemo.asr import load_model
from utils import get_voice_files, get_video_files

# ReazonSpeech model for Speech-to-Text
# See: https://huggingface.co/reazon-research/reazonspeech-nemo-v2
REAZON_MODEL = load_model(device="cuda")


def _get_subject_segments(audio, min_silence_len=1000, silence_thresh=-50):
    """
    音声データから音のある区間（被験者の区間）の開始ミリ秒・終了ミリ秒を取得
    """
    logger.info("Detecting subject segments...")
    # 音のある区間を抽出する
    nonsilent_segments = detect_nonsilent(
        audio,
        # 以下のパラメータによって出力されるテキストの長さが変わる
        min_silence_len=min_silence_len,  # min_silence_len ミリ秒以上無音なら区間を抽出
        silence_thresh=silence_thresh,  # slice_thresh dBFS以下で無音とみなす
    )
    subject_segments = []
    for start, end in nonsilent_segments:
        if end - start < 1000:  # 1秒未満の区間は無視する
            logger.info(f"Ignoring segment of {end - start} ms")
            continue
        subject_segments.append((start, end))
    logger.info("Successfully get subject segments!")
    return subject_segments


def _get_subject_text_list(voice_file_path, start, end):
    """
    発話開始秒, 発話終了秒, 発話テキスト からなるCSV行を生成する
    """
    audio = audio_from_path(voice_file_path)
    try:
        logger.info(
            f"Transcribing segment from {start} ms to {end} ms in {voice_file_path}..."
        )
        result = transcribe(REAZON_MODEL, audio)
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None
    text = result.text
    if len(text) > 0:
        return [start / 1000, end / 1000, text]  # ミリ秒を秒に直してからCSVに書き込む
    return None


def _get_subject_voice_text(
    audio, subject_segments, voice_output_file_path, text_output_file_path
):
    """
    音のある区間（被験者の区間）だけ音声データを抜き出す
    """
    subject_voice_sum = AudioSegment.empty()
    subject_text_list = []
    logger.info("Extracting subject audio and text...")
    utterance_count = 1
    for start, end in subject_segments:
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
    # 被験者の区間のみの音声データを保存
    subject_voice_sum.export(voice_output_file_path, format="wav")
    logger.info(f"Successfully get subject audio at {voice_output_file_path}!")
    # 被験者の発話テキストを保存
    with open(text_output_file_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_seconds", "end_seconds", "text"])
        writer.writerows(subject_text_list)
    logger.info(f"Successfully get subject text at {text_output_file_path}!")
    return


def _get_subject_video(video_file, subject_segments, video_output_file_path):
    """
    subject_segmentsを使って、被験者の映っている区間の動画フレームを抜き出す
    """
    video = VideoFileClip(video_file)
    # 動画の切り抜きと保存
    clips = []
    for i, (start_ms, end_ms) in enumerate(subject_segments):
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
    logger.info(f"Successfully get subject video at {video_output_file_path}!")
    return


def _preprocess(video_file_path, voice_file_path, save_dir, text_output_file_path):
    """
    前処理を行う
    """
    logger.info(f"Start preprocessing from {video_file_path} and {voice_file_path}...")
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(voice_file_path)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = _get_subject_segments(audio)

    voice_output_file_name = os.path.splitext(os.path.basename(voice_file_path))[0]
    voice_output_file_path = os.path.join(save_dir, f"{voice_output_file_name}.wav")
    # subject_segmentsを利用して音声データから被験者の音声と発話テキストを抜き出す
    _get_subject_voice_text(
        audio,
        subject_segments,
        voice_output_file_path,
        text_output_file_path,
    )

    video_output_file_name = os.path.splitext(os.path.basename(video_file_path))[0]
    video_output_file_path = os.path.join(save_dir, f"{video_output_file_name}.mp4")
    # subject_segmentsを利用して動画データから被験者の動画フレームを抜き出す
    _get_subject_video(video_file_path, subject_segments, video_output_file_path)
    return


def main(input_data_dir, output_data_dir):
    logger.info("Start preprocessing...")
    os.makedirs(output_data_dir, exist_ok=True)

    voice_files = get_voice_files(input_data_dir)
    video_files = get_video_files(input_data_dir)

    if len(voice_files) != len(video_files):
        logger.error(
            f"voice_files: {len(voice_files)} != video_files: {len(video_files)}"
        )
        raise ValueError("voice_filesとvideo_filesの数が一致しません")

    for voice_file, video_file in zip(voice_files, video_files):
        voice_data_id = voice_file[0]
        voice_file_path = voice_file[1]
        video_data_id = video_file[0]
        video_file_path = video_file[1]
        logger.info(f"voice_data_id: {voice_data_id}")
        logger.info(f"video_data_id: {video_data_id}")
        logger.info(f"voice_file_path: {voice_file_path}")
        logger.info(f"video_file_path: {video_file_path}")

        if voice_data_id != video_data_id:
            logger.error(
                f"voice_data_id: {voice_data_id} != video_data_id: {video_data_id}"
            )
            raise ValueError("voice_data_idとvideo_data_idが一致しません")

        data_id = voice_data_id
        multimodal_save_dir = os.path.join(output_data_dir, data_id)
        os.makedirs(multimodal_save_dir, exist_ok=True)
        os.makedirs(os.path.join(output_data_dir, "text_elan"), exist_ok=True)
        text_output_file_path = os.path.join(
            output_data_dir, "text_elan", f"{data_id}.csv"
        )
        _preprocess(
            video_file_path, voice_file_path, multimodal_save_dir, text_output_file_path
        )
    logger.info("Finished preprocessing!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_dir",
        help="Path to input data directory",
        type=str,
        default="../data/raw",
    )

    parser.add_argument(
        "--output_data_dir",
        default="../data/preprocessed",
        help="Path to output directory",
    )

    args = parser.parse_args()
    input_data_dir = args.input_data_dir
    output_data_dir = args.output_data_dir

    logger.info(f"Input data dir: {input_data_dir}")
    logger.info(f"Output data dir: {output_data_dir}")
    main(input_data_dir, output_data_dir)
