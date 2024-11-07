import torch
import os
from logzero import logger
from utils import save_feature, get_igaku_target, get_riko_target, get_voice_files

# VGGishのPyTorch実装
# See: https://github.com/harritaylor/torchvggish
model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.eval()


def extract_vggish_features(input_data_dir):
    riko_voice_files, igaku_voice_files = get_voice_files(input_data_dir)

    for riko_voice_file in riko_voice_files:
        logger.info(f"Extracting VGGish features from {riko_voice_file}....")
        # VGGishの特徴量を取得
        feature = model.forward(riko_voice_file)
        data_id = riko_voice_file.split("/")[-2]
        target = get_riko_target(data_id)
        save_feature(
            feature.detach().cpu().numpy(),
            os.path.join(input_data_dir, "vggish"),
            target,
        )

    for igaku_voice_file in igaku_voice_files:
        logger.info(f"Extracting VGGish features from {igaku_voice_file}....")
        # VGGishの特徴量を取得
        feature = model.forward(igaku_voice_file)
        data_id = igaku_voice_file.split("/")[-2]
        target = get_igaku_target(data_id)
        save_feature(
            feature.detach().cpu().numpy(),
            os.path.join(input_data_dir, "vggish"),
            target,
        )

    return
