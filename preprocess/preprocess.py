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
    subject_segments = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    # 取得した区間をpickleで保存
    if output_dir:
        with open(os.path.join(output_dir, "subject_segments.pickle"), mode="wb") as f:
            pickle.dump(subject_segments, f)
    logger.info("Successfully get subject segments!")
    return subject_segments


def _remove_tmp_file(tmp_file_path):
    try:
        os.remove(tmp_file_path)
    except OSError as e:
        logger.error(f"Failed to remove file: {e}")


def _get_subject_text(tmp_audio_file_path):
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(tmp_audio_file_path, "rb"),
        language="ja",
        timeout=300,
    )
    text = transcription.text
    if len(text) > 1:  # 2文字以上の発話のみ書き込む
        return text.strip() + "\n"
    return ""


# 音のある区間（被験者の区間）だけ音声データを抜き出す
def get_subject_audio_text(
    audio, subject_segments, audio_output_file_path, text_output_file_path
):
    subject_audio_sum = AudioSegment.empty()
    subject_text = ""
    for start, end in subject_segments:
        if end - start > 300:  # 発話区間が0.3s以上のとき抜き出す
            subject_audio_sum += audio[start:end]
            subject_audio = AudioSegment.empty()
            subject_audio += audio[start:end]
            subject_audio.export("tmp.mp3", format="mp3")
            subject_text += _get_subject_text("tmp.mp3")
            _remove_tmp_file("tmp.mp3")
    # 被験者の区間のみの音声データを保存
    subject_audio_sum.export(audio_output_file_path, format="mp3")
    with open(text_output_file_path, mode="w", encoding="utf-8") as f:
        f.write(subject_text)
    logger.info("Successfully get subject audio and text!")
    return


# カウンセリングの動画データから被験者のみのフレームを取得する
def get_subject_frames(video_file, subject_segments, video_output_file_path):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    subject_frames = []
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(
        video_output_file_path,
        fourcc,
        frame_rate,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

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
    logger.info("Successfully get subject frames!")


def main(video_file_path, audio_file_path, output_dir, faculty):
    make_processed_data_dir(output_dir)
    # pydubで音声ファイルを開く
    audio = AudioSegment.from_file(audio_file_path)
    # 音声データから被験者が喋っている区間のミリ秒を取得する
    subject_segments = get_subject_segments(audio)
    # ↑を利用して音声データから被験者の音声データを抜き出す
    audio_output_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    audio_output_file_path = os.path.join(
        output_dir, "voice", faculty, f"{audio_output_file_name}.mp3"
    )
    text_output_file_path = os.path.join(
        output_dir, "text", faculty, f"{audio_output_file_name}.txt"
    )
    get_subject_audio_text(
        audio, subject_segments, audio_output_file_path, text_output_file_path
    )
    video_output_file_path = os.path.join(
        output_dir, "video", faculty, "subject_video.mp4"
    )
    get_subject_frames(
        video_file_path, subject_segments, video_output_file_path, output_dir
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video_path", help="Path to input video file")
    parser.add_argument("input_audio_path", help="Path to input audio file")
    parser.add_argument(
        "--output_dir",
        default="processed_data",
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
    video_file_path = args.input_video_path
    audio_file_path = args.input_audio_path
    output_dir = args.output_dir
    faculty = args.faculty

    logger.info(f"Input video: {video_file_path}")
    logger.info(f"Input audio: {audio_file_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Data of Faculty: {faculty}")
    main(video_file_path, audio_file_path, output_dir, faculty)
