import csv
from logzero import logger
import sys
from src.extract import extract_feature

OUTPUT_DIR = "output"


def main(filename):
    if not filename.endswith(".csv"):
        logger.error("Input file must be a csv file.")
        return
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if len(row) != 2:
                logger.warn("Skip line: " + row)
                continue
            if not row[0].endswith(".mp4") or not row[1].endswith(".m4a"):
                logger.warn("Skip line: " + row)
                continue
            video_file_path = row[0].strip()
            audio_file_path = row[1].strip()
            extract_feature(video_file_path, audio_file_path, f"{OUTPUT_DIR}/0{i+1}")
            logger.info(f"Extracted feature from {row[0]} and {row[1]}.")


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        logger.error("Usage: python3 main.py input.csv")
    else:
        main(args[1])
