from dotenv import load_dotenv
from os.path import join, dirname
import os
from logzero import logger


def load_env():
    load_dotenv(verbose=True)
    dotenv_path = join(dirname(__file__), ".env")
    load_dotenv(dotenv_path)


def remove_tmp_file(tmp_file_path):
    try:
        os.remove(tmp_file_path)
    except OSError as e:
        logger.error(f"Failed to remove file: {e}")


def make_dir(dir_name):
    # TODO: TemporaryDirectoryで書き換える
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_processed_data_dir(output_dir):
    for modal in ["text", "voice", "video"]:
        for faculty in ["igaku", "riko"]:
            os.makedirs(f"{output_dir}/{modal}/{faculty}", exist_ok=True)
    logger.info("Successfully make processed data directory!")
    return
