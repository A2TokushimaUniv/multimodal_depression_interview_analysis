from pydub.silence import detect_nonsilent
import os
from pydub import AudioSegment
from logzero import logger
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips
import csv
from reazonspeech.nemo.asr import transcribe, audio_from_path
import glob
from reazonspeech.nemo.asr import load_model

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
        # NOTE: 以下のパラメータによって出力されるテキストの長さが変わる
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


def get_subject_text_list(audio_file, start, end):
    """
    発話開始秒, 発話終了秒, 発話テキスト からなるCSV行を生成する
    """
    audio = audio_from_path(audio_file)
    try:
        logger.info(
            f"Transcribing segment from {start} ms to {end} ms in {audio_file}..."
        )
        result = transcribe(REAZON_MODEL, audio)
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None
    text = result.text
    if len(text) > 0:
        return [start / 1000, end / 1000, text]  # ミリ秒を秒に直してからCSVに書き込む
    return None


def _get_subject_audio_text(
    audio, subject_segments, audio_output_dir, audio_output_file, text_output_file
):
    """
    音のある区間（被験者の区間）だけ音声データを抜き出す
    """
    subject_audio_sum = AudioSegment.empty()
    subject_text_list = []
    logger.info("Extracting subject audio and text...")
    utterance_count = 1
    for start, end in subject_segments:
        subject_audio_sum += audio[start:end]
        subject_audio = AudioSegment.empty()
        subject_audio += audio[start:end]
        utterance_audio_path = os.path.join(
            audio_output_dir, f"utterance_{utterance_count}.wav"
        )
        # 発話ごとに音声を保存
        subject_audio.export(utterance_audio_path, format="wav")
        # 発話ごとのテキストを抽出
        text_list = get_subject_text_list(utterance_audio_path, start, end)
        if text_list:
            subject_text_list.append(text_list)
        utterance_count += 1
    # 被験者の区間のみの音声データを保存
    subject_audio_sum.export(audio_output_file, format="wav")
    logger.info(f"Successfully get subject audio at {audio_output_file}!")
    # 被験者の発話テキストを保存
    with open(text_output_file, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_seconds", "end_seconds", "text"])
        writer.writerows(subject_text_list)
    logger.info(f"Successfully get subject text at {text_output_file}!")
    return


def _get_subject_video(
    video_file, subject_segments, video_output_dir, video_output_file
):
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
        clip_filename = f"utterance_{i+1}.mp4"
        clip.write_videofile(
            os.path.join(video_output_dir, clip_filename), codec="libx264"
        )
        clips.append(clip)
    # 切り抜いた動画を結合
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(video_output_file, codec="libx264")

    # リソースを解放
    video.reader.close()
    video.audio.reader.close_proc()
    for clip in clips:
        clip.reader.close()
        clip.audio.reader.close_proc()
    logger.info(f"Successfully get subject video at {video_output_file}!")
    return


def _make_processed_data_dir(output_dir, faculty, dir_num):
    """
    前処理後のデータを格納するディレクトリを作成
    """
    for modal in ["text", "voice", "video"]:
        os.makedirs(os.path.join(output_dir, modal, faculty, dir_num), exist_ok=True)
    return


def _preprocess(video_file, audio_file, output_dir, faculty, dir_num):
    """
    前処理を行う
    """
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = _get_subject_segments(audio)

    audio_output_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    audio_output_dir = os.path.join(output_dir, "voice", faculty, dir_num)
    audio_output_file = os.path.join(audio_output_dir, f"{audio_output_file_name}.wav")
    text_output_file = os.path.join(
        output_dir, "text", faculty, dir_num, f"{audio_output_file_name}.csv"
    )
    # subject_segmentsを利用して音声データから被験者の音声と発話テキストを抜き出す
    _get_subject_audio_text(
        audio,
        subject_segments,
        audio_output_dir,
        audio_output_file,
        text_output_file,
    )

    video_output_file_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_dir = os.path.join(output_dir, "video", faculty, dir_num)
    video_output_file = os.path.join(video_output_dir, f"{video_output_file_name}.mp4")
    # subject_segmentsを利用して動画データから被験者の動画フレームを抜き出す
    _get_subject_video(
        video_file, subject_segments, video_output_dir, video_output_file
    )
    return


def main(input_data_dir, output_dir):
    logger.info("Start preprocessing...")
    riko_audio_files = sorted(
        glob.glob(
            os.path.join(
                input_data_dir, "voice", "riko", "*", "audioNLP*.m4a"
            ),  # 被験者の音声を読み込み
            recursive=True,
        )
    )
    riko_video_files = sorted(
        glob.glob(
            os.path.join(input_data_dir, "video", "riko", "*", "*.mp4"),
            recursive=True,
        )
    )
    igaku_audio_files = sorted(
        glob.glob(
            os.path.join(
                input_data_dir,
                "voice",
                "igaku",
                "*",
                "*_zoom_音声_被験者*.m4a",  # 被験者の音声を読み込み
            ),
            recursive=True,
        )
    )
    igaku_video_files = sorted(
        glob.glob(
            os.path.join(input_data_dir, "video", "igaku", "*", "*.mp4"),
            recursive=True,
        )
    )

    # 被験者の音声と対応する動画をペアにして前処理を行う
    logger.info("Start preprocessing riko raw data.")
    for riko_audio, riko_video in zip(riko_audio_files, riko_video_files):
        dir_num = os.path.basename(os.path.dirname(riko_audio))
        # 前処理後の結果を格納するディレクトリを作成
        _make_processed_data_dir(output_dir, "riko", dir_num)
        _preprocess(riko_video, riko_audio, output_dir, "riko", dir_num)

    logger.info("Start preprocessing igaku raw data.")
    for igaku_audio, igaku_video in zip(igaku_audio_files, igaku_video_files):
        dir_num = os.path.basename(os.path.dirname(igaku_audio))
        _make_processed_data_dir(output_dir, "igaku", dir_num)
        _preprocess(igaku_video, igaku_audio, output_dir, "igaku", dir_num)
    logger.info("Finished preprocessing!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_dir",
        help="Path to input data directory",
        type=str,
        default="../data/raw_data",
    )

    parser.add_argument(
        "--output_dir",
        default="../data/preprocessed_data",
        help="Path to output directory",
    )

    args = parser.parse_args()
    input_data_dir = args.input_data_dir
    output_dir = args.output_dir

    logger.info(f"Input dir: {input_data_dir}")
    logger.info(f"Output dir: {output_dir}")
    main(input_data_dir, output_dir)
