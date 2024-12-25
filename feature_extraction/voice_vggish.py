import pandas as pd
import torch
import os
from logzero import logger
from utils import save_feature, get_voice_files

# VGGishのPyTorch実装
# See: https://github.com/harritaylor/torchvggish
model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.eval()


def extract_vggish_feature(input_data_dir, output_data_dir):
    """
    VGGishの特徴量を抽出する
    """
    logger.info("VGGishの特徴量を抽出しています....")
    voice_files = get_voice_files(input_data_dir)

    for data_id, voice_file in voice_files:
        logger.info(f"{voice_file}からVGGishの特徴量を抽出しています....")
        # VGGishの特徴量を取得
        feature = model.forward(voice_file)
        save_feature(
            pd.DataFrame(feature.detach().cpu().numpy()),
            os.path.join(output_data_dir, "vggish"),
            f"{data_id}.csv",
        )
    logger.info("VGGishの特徴量を抽出しました")
    return
