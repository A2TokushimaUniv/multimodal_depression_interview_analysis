import numpy as np
import pandas as pd
from utils import save_feature, get_riko_target, get_igaku_target
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import glob
import os
from logzero import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# See: https://huggingface.co/facebook/wav2vec2-large-xlsr-53
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
model.eval()


def _input_wav2vec2(voice_file):
    """
    ある音声ファイルのwav2vec2の特徴量を取得する
    """
    # 入力音声は16kHzでサンプリングする必要がある
    input_audio, sample_rate = librosa.load(voice_file, sr=16000)

    # 音声が短すぎる場合や無音の場合をチェック
    if len(input_audio) < feature_extractor.sampling_rate:  # 1秒未満
        logger.warning(f"Skipping {voice_file} due to insufficient length.")
        return None

    # 無音をチェック（例として、音量が非常に小さい場合を無音と見なす）
    if np.max(np.abs(input_audio)) < 1e-5:  # 許容される音量の閾値
        logger.warning(f"Skipping {voice_file} due to silence.")
        return None

    i = feature_extractor(
        input_audio, return_tensors="pt", sampling_rate=sample_rate
    ).to(device)
    with torch.no_grad():
        o = model(i.input_values)
    return o.last_hidden_state.squeeze(0)


def _get_wav2vec2_feature(voice_files):
    """
    ある被験者のwav2vec2の特徴量を取得する
    """
    results = []
    for voice_file in voice_files:
        wav2vec2_feature = _input_wav2vec2(voice_file)
        if wav2vec2_feature is not None:
            results.append(wav2vec2_feature)
    return torch.cat(results, dim=0)


def extract_wav2vec2_feature(input_data_dir):
    riko_voice_dirs = os.listdir(os.path.join(input_data_dir, "voice", "riko"))
    for riko_voice_dir in riko_voice_dirs:
        target = get_riko_target(riko_voice_dir)
        # 音声ファイルを一度に入力するとメモリが足りなくなるので、一発話ずつ入力する
        riko_voice_files = glob.glob(
            os.path.join(
                input_data_dir, "voice", "riko", riko_voice_dir, "utterance_*.wav"
            ),
            recursive=True,
        )
        riko_voice_files.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        logger.info(
            f"Extracting Wav2Vec2 feature from voice files in voice/riko/{riko_voice_dir}...."
        )
        feature = _get_wav2vec2_feature(riko_voice_files)
        save_feature(
            pd.DataFrame(feature.detach().cpu().numpy()),
            os.path.join(input_data_dir, "wav2vec2"),
            target,
        )
    igaku_voice_dirs = os.listdir(os.path.join(input_data_dir, "voice", "igaku"))
    for igaku_voice_dir in igaku_voice_dirs:
        target = get_igaku_target(igaku_voice_dir)
        # 音声ファイルを一度に入力するとメモリが足りなくなるので、一発話ずつ入力する
        igaku_voice_files = glob.glob(
            os.path.join(
                input_data_dir, "voice", "igaku", igaku_voice_dir, "utterance_*.wav"
            ),
            recursive=True,
        )
        igaku_voice_files.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        logger.info(
            f"Extracting Wav2Vec2 feature from voice files in voice/riko/{igaku_voice_dir}...."
        )
        feature = _get_wav2vec2_feature(igaku_voice_files)
        save_feature(
            pd.DataFrame(feature.detach().cpu().numpy()),
            os.path.join(input_data_dir, "wav2vec2"),
            target,
        )
    return
