from pydub.silence import detect_nonsilent
from .utils import load_dotenv
import os
from openai import OpenAI
from pydub import AudioSegment
import cv2
import pickle
from logzero import logger

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# 音声データから音のある区間（被験者の区間）の開始ミリ秒・終了ミリ秒を取得
def get_subject_segments(audio, output_dir, min_silence_len=500, silence_thresh=-50):
    subject_segments = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    # 取得した区間をpickleで保存
    with open(os.path.join(output_dir, "subject_segments.pickle"), mode="wb") as f:
        pickle.dump(subject_segments, f)
    logger.info("Successfully get subject segments!")
    return subject_segments


# 音のある区間（被験者の区間）だけ音声データを抜き出す
def get_subject_audio(audio, subject_segments, output_dir):
    subject_audio = AudioSegment.empty()
    for start, end in subject_segments:
        subject_audio += audio[start:end]
    # 被験者の区間のみの音声データを保存
    # TODO: m4aでなくてもいい？mp3のほうが使いやすい
    subject_audio_file = os.path.join(output_dir, "subject_audio.mp3")
    subject_audio.export(subject_audio_file, format="mp3")
    logger.info("Successfully get subject audio!")
    return subject_audio_file


# カウンセリングの動画データからカ被験者のみのフレームを取得する
def get_subject_frames(video_file, subject_segments, output_dir):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    subject_frames = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    subject_video_file = os.path.join(output_dir, "subject_video.mp4")
    out = cv2.VideoWriter(
        subject_video_file,
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

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Successfully get subject frames!")
    return subject_frames


# 被験者の音声データをテキスト化する
def get_subject_text(subject_audio_file, output_dir):
    # OpenAI API Whisperで音をテキストに変換する
    # TODO: OpenAI API経由でないほうが良い？
    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_file = open(subject_audio_file, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    subject_text = transcription.text
    # 被験者のテキストを保存
    with open(
        os.path.join(output_dir, "subject_text.txt"), encoding="utf-8", mode="w"
    ) as f:
        f.write(subject_text)

    logger.info("Successfully get subject text!")
    return subject_text
