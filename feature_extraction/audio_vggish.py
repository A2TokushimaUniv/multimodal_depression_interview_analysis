import torch
import os
from logzero import logger
import numpy as np
from utils import save_as_npy, get_igaku_target, get_riko_target
import glob

# VGGishのPyTorch実装
# See: https://github.com/harritaylor/torchvggish
model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.eval()


def _save_vggish_feature(feature, output_dir, target):
    vggish_dir = "vggish"
    os.makedirs(os.path.join(output_dir, vggish_dir), exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, vggish_dir, f"{target}.csv"),
        feature.detach().cpu().numpy(),  # CPUに移動してからnumpyに変換する
        delimiter=",",
    )
    save_as_npy(
        os.path.join(output_dir, vggish_dir, f"{target}.csv"),
        os.path.join(output_dir, f"{vggish_dir}_npy"),
        skip_header=False,
    )


def extract_vggish_features(input_data_dir):
    riko_audio_files = glob.glob(
        os.path.join(input_data_dir, "voice", "riko", "*", "audioNLP*.wav"),
        recursive=True,
    )
    igaku_audio_files = glob.glob(
        os.path.join(input_data_dir, "voice", "igaku", "*", "*_zoom_音声_被験者*.wav"),
        recursive=True,
    )

    for riko_audio_file in riko_audio_files:
        logger.info(f"Extracting VGGish features from {riko_audio_file}....")
        feature = model.forward(riko_audio_file)
        data_id = riko_audio_file.split("/")[-2]
        target = get_riko_target(data_id)
        _save_vggish_feature(feature, input_data_dir, target)

    for igaku_audio_file in igaku_audio_files:
        logger.info(f"Extracting VGGish features from {igaku_audio_file}....")
        feature = model.forward(igaku_audio_file)
        data_id = igaku_audio_file.split("/")[-2]
        target = get_igaku_target(data_id)
        _save_vggish_feature(feature, input_data_dir, target)

    return
