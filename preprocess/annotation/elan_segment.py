"""
# ELANでのアノテーションのための被験者の音声ファイルから、
# 発話区間と発話からなるCSVを出力するスクリプト
"""

from pydub.silence import detect_nonsilent
from openai import OpenAI
import argparse
from logzero import logger
from pydub import AudioSegment
import os
import csv

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def _remove_tmp_file(tmp_file_path):
    try:
        os.remove(tmp_file_path)
    except OSError as e:
        logger.error(f"Failed to remove file: {e}")


def main(audio_file, output_file):
    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_file = open(audio_file, "rb")
    try:
        audio = AudioSegment.from_file(audio_file, format="m4a")
    except Exception as e:
        logger.error("Failed to load audio file!")
        logger.error(f"Exception: {e}")
        return

    logger.info("Detecting non-silent segments...")
    subject_segments = detect_nonsilent(
        # ここのパラメータによって出力されるテキストの長さが変わる
        audio,
        min_silence_len=3000,  # min_silence_len ミリ秒以上無音なら区間を抽出
        silence_thresh=-50,  # slice_thresh dBFS以下で無音とみなす
    )

    for start, end in subject_segments:
        if end - start > 300:  # 発話区間が0.3s以上のとき抜き出す
            subject_audio = AudioSegment.empty()
            subject_audio += audio[start:end]
            subject_audio.export("tmp.mp3", format="mp3")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=open("tmp.mp3", "rb"),
                language="ja",
                timeout=300,
            )
            text = transcription.text
            if len(text) > 1:  # 2文字以上の発話のみ書き込む
                # ミリ秒を秒に直してからCSVに書き込む
                line = [start / 1000, end / 1000, text]
                logger.info("Adding line to CSV file: {}".format(line))
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
            _remove_tmp_file("tmp.mp3")
    logger.info("Successfully generated ELAN CSV file: {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser
    parser.add_argument("input_audio_file", help="Input audio file path")
    parser.add_argument("output_csv_file", help="Output CSV file path")
    args = parser.parse_args()

    input_audio_file = args.input_audio_file
    output_csv_file = args.input_csv_file
    logger.info(f"Input audio file: {input_audio_file}")
    logger.info(f"Output CSV file: {output_csv_file}")
    main(input_audio_file, output_csv_file)
