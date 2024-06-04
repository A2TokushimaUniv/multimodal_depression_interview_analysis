from pydub.silence import detect_nonsilent
from utils import load_dotenv, make_processed_data_dir
import os
from openai import OpenAI
from pydub import AudioSegment
import cv2
import pickle
from logzero import logger
import argparse

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
        if end - start > 300:  # 発話区間が0.3s以上のとき抜き出す
            subject_segments.append([start, end])
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
        utterance_audio_path = f"{audio_output_dir}/utterance{utterance_count}.mp3"
        # 発話ごとに音声を保存
        subject_audio.export(utterance_audio_path, format="mp3")
        # 発話ごとのテキストを抽出
        subject_text += _get_subject_text(utterance_audio_path)
    # 被験者の区間のみの音声データを保存
    subject_audio_sum.export(audio_output_file, format="mp3")
    logger.info(f"Successfully get subject audio at {audio_output_file}!")
    with open(text_output_file, mode="w", encoding="utf-8") as f:
        f.write(subject_text)
    logger.info(f"Successfully get subject text at {text_output_file}!")
    return


# カウンセリングの動画データから被験者のみのフレームを取得する
# TODO: ここ多分バグってるな、、
def get_subject_frames(
    video_file, subject_segments, video_output_dir, video_output_file
):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    subject_frames = []
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(
        video_output_file,
        fourcc,
        frame_rate,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    logger.info("Extracting subject frames...")
    # TODO: 発話ごとに動画を保存する
    for start, end in subject_segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * frame_rate))

        while True:
            ret, frame = cap.read()
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = current_frame / frame_rate

            if current_time > end or not ret:
                break

            subject_frames.append(frame)
            out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Successfully get subject frames at {video_output_file}!")


def main(video_file, audio_file, output_dir, faculty):
    make_processed_data_dir(output_dir)
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = get_subject_segments(audio)
    # ↑を利用して音声データから被験者の音声データを抜き出す
    audio_output_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    audio_output_dir = os.path.join(output_dir, "voice", faculty)
    audio_output_file = os.path.join(
        f"{audio_output_dir}", f"{audio_output_file_name}.mp3"
    )
    text_output_file = os.path.join(
        output_dir, "text", faculty, f"{audio_output_file_name}.txt"
    )
    get_subject_audio_text(
        audio, subject_segments, audio_output_dir, audio_output_file, text_output_file
    )

    video_output_file_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_dir = os.path.join(output_dir, "video", faculty)
    video_output_file = os.path.join(
        f"{video_output_dir}", f"{video_output_file_name}.mp4"
    )
    get_subject_frames(
        video_file, subject_segments, video_output_dir, video_output_file
    )
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

    args = parser.parse_args()
    video_file = args.input_video
    audio_file = args.input_audio
    output_dir = args.output_dir
    faculty = args.faculty

    logger.info(f"Input video: {video_file}")
    logger.info(f"Input audio: {audio_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Data of Faculty: {faculty}")
    main(video_file, audio_file, output_dir, faculty)
