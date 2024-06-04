from pydub.silence import detect_nonsilent
from utils import load_dotenv, make_processed_data_dir
import os
from openai import OpenAI
from pydub import AudioSegment
import pickle
from logzero import logger
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# 音声データから音のある区間（被験者の区間）の開始ミリ秒・終了ミリ秒を取得
def get_subject_segments(
    audio, output_dir=None, min_silence_len=3000, silence_thresh=-50
):
    logger.info("Detecting subject segments...")
    nonsilent_segments = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,  # min_silence_len ミリ秒以上無音なら区間を抽出
        silence_thresh=silence_thresh,  # slice_thresh dBFS以下で無音とみなす
    )
    subject_segments = []
    for start, end in nonsilent_segments:
        if end - start > 500:  # 発話区間が0.3s以上のとき抜き出す
            subject_segments.append((start, end))
    # 取得した区間をpickleで保存
    if output_dir:
        with open(os.path.join(output_dir, "subject_segments.pickle"), mode="wb") as f:
            pickle.dump(subject_segments, f)
    logger.info("Successfully get subject segments!")
    return subject_segments


def _get_subject_text(audio_file):
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_file, "rb"),
        language="ja",
        timeout=300,
    )
    text = transcription.text
    if len(text) > 1:  # 2文字以上の発話のみ書き込む
        return text.strip() + "\n"
    return ""


# 音のある区間（被験者の区間）だけ音声データを抜き出す
def get_subject_audio_text(
    audio, subject_segments, audio_output_dir, audio_output_file, text_output_file
):
    subject_audio_sum = AudioSegment.empty()
    subject_text = ""
    logger.info("Extracting subject audio and text...")
    utterance_count = 1
    for start, end in subject_segments:
        subject_audio_sum += audio[start:end]
        subject_audio = AudioSegment.empty()
        subject_audio += audio[start:end]
        utterance_audio_path = os.path.join(
            audio_output_dir, f"utterance_{utterance_count}.mp3"
        )
        # 発話ごとに音声を保存
        subject_audio.export(utterance_audio_path, format="mp3")
        # 発話ごとのテキストを抽出
        subject_text += _get_subject_text(utterance_audio_path)
        utterance_count += 1
    # 被験者の区間のみの音声データを保存
    subject_audio_sum.export(audio_output_file, format="mp3")
    logger.info(f"Successfully get subject audio at {audio_output_file}!")
    with open(text_output_file, mode="w", encoding="utf-8") as f:
        f.write(subject_text)
    logger.info(f"Successfully get subject text at {text_output_file}!")
    return


def get_subject_video(
    video_file, subject_segments, video_output_dir, video_output_file
):
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


def main(video_file, audio_file, output_dir, faculty, dir_num):
    make_processed_data_dir(output_dir, dir_num)
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = get_subject_segments(audio)
    # ↑を利用して音声データから被験者の音声データを抜き出す
    audio_output_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    audio_output_dir = os.path.join(output_dir, "voice", faculty, dir_num)
    audio_output_file = os.path.join(audio_output_dir, f"{audio_output_file_name}.mp3")
    text_output_file = os.path.join(
        output_dir, "text", faculty, dir_num, f"{audio_output_file_name}.txt"
    )
    get_subject_audio_text(
        audio, subject_segments, audio_output_dir, audio_output_file, text_output_file
    )

    video_output_file_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_dir = os.path.join(output_dir, "video", faculty, dir_num)
    video_output_file = os.path.join(video_output_dir, f"{video_output_file_name}.mp4")
    get_subject_video(video_file, subject_segments, video_output_dir, video_output_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", help="Path to input video file", required=True)
    parser.add_argument("--input_audio", help="Path to input audio file", required=True)
    parser.add_argument(
        "--output_dir",
        default="preprocessed_data",
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--faculty",
        choices=["riko", "igaku"],
        required=True,
        help="Faculty of the subject",
    )
    parser.add_argument("--dir_num", default="0", help="Directory number")

    args = parser.parse_args()
    video_file = args.input_video
    audio_file = args.input_audio
    output_dir = args.output_dir
    faculty = args.faculty
    dir_num = args.dir_num

    logger.info(f"Input video: {video_file}")
    logger.info(f"Input audio: {audio_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Data of Faculty: {faculty}")
    logger.info(f"Directory number: {dir_num}")
    main(video_file, audio_file, output_dir, faculty, dir_num)
