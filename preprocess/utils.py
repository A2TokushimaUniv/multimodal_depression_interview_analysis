from dotenv import load_dotenv
from os.path import join, dirname
import os
from logzero import logger


def load_env():
    load_dotenv(verbose=True)
    dotenv_path = join(dirname(__file__), ".env")
    load_dotenv(dotenv_path)


def make_processed_data_dir(output_dir, dir_num):
    for modal in ["text", "voice", "video"]:
        for faculty in ["igaku", "riko"]:
            os.makedirs(
                os.path.join(output_dir, modal, faculty, dir_num), exist_ok=True
            )
    logger.info("Successfully make processed data directory!")
    return
