from pydub.silence import detect_nonsilent
from openai import OpenAI
import sys
from logzero import logger
from pydub import AudioSegment
import os
import csv

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def remove_tmp_file(tmp_file_path):
    try:
        os.remove(tmp_file_path)
    except OSError as e:
        logger.error(f"Failed to remove file: {e}")


def main(audio_file, output_file):
    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_file = open(audio_file, "rb")
    audio = AudioSegment.from_file(audio_file)
    subject_segments = subject_segments = detect_nonsilent(
        # ここのパラメータによって出力されるテキストの長さが変わる
        audio,
        min_silence_len=3000,  # min_silence_lenミリ秒以上無音なら区間を抽出
        silence_thresh=-50,  #
    )

    for start, end in subject_segments:
        if end - start > 300:  # 発話区間は0.3s以上のとき
            subject_audio = AudioSegment.empty()
            subject_audio += audio[start:end]
            subject_audio.export("tmp.mp3", format="mp3")
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=open("tmp.mp3", "rb"), language="ja"
            )
            text = transcription.text
            if len(text) > 1:
                line = [start / 1000, end / 1000, text]
                logger.info(line)
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
            remove_tmp_file("tmp.mp3")


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        # 処理したい音声ファイル名と出力ファイル名を指定して実行する
        logger.error(
            "Usage: python3 preprocess/annotation_speech_segment.py audio.m4a output.csv"
        )
    else:
        main(args[1], args[2])
