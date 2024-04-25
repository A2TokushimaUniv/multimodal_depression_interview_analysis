from pydub.silence import detect_nonsilent
from .utils import load_dotenv
import os
from openai import OpenAI
from pydub import AudioSegment
import cv2

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# 音声データから音のある区間（被験者の区間）の開始ミリ秒・終了ミリ秒を取得
def get_subject_timestamp(audio, min_silence_len=500, silence_thresh=-50):
    subject_segments = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    return subject_segments


# 音のある区間（被験者の区間）だけ音声データを抜き出す
def get_subject_audio(audio, subject_segments):
    subject_audio = AudioSegment.empty()
    for start, end in subject_segments:
        subject_audio += audio[start:end]
    return subject_audio


# カウンセリングの動画データからカ被験者のみのフレームを取得する
def get_subject_frames(movie_file, subject_segments):
    cap = cv2.VideoCapture(movie_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    subject_frames = []

    for start, end in subject_segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * frame_rate))

        while True:
            ret, frame = cap.read()
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = current_frame / frame_rate

            if current_time > end or not ret:
                break

            subject_frames.append(frame)

    cap.release()
    return subject_frames


# 被験者の音声データをテキスト化する
def get_subject_text(subject_audio_file):
    # OpenAI API Whisperで音をテキストに変換する
    # TODO: OpenAI API経由でないほうが良い？
    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_file = open(subject_audio_file, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return transcription.text
