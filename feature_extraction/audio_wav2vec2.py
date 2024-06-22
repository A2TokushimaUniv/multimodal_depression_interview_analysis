import numpy as np
from utils import save_as_npy
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


def _input_wav2vec2(audio_file):
    """
    ある音声ファイルのwav2vec2の特徴量を取得する
    """
    logger.info(f"Extracting Wav2Vec2 features from {audio_file}....")
    # 入力音声は16kHzでサンプリングする必要がある
    input_audio, sample_rate = librosa.load(audio_file, sr=16000)

    # 音声が短すぎる場合や無音の場合をチェック
    if len(input_audio) < feature_extractor.sampling_rate:  # 1秒未満
        logger.warning(f"Skipping {audio_file} due to insufficient length.")
        return None

    # 無音をチェック（例として、音量が非常に小さい場合を無音と見なす）
    if np.max(np.abs(input_audio)) < 1e-5:  # 許容される音量の閾値
        logger.warning(f"Skipping {audio_file} due to silence.")
        return None

    i = feature_extractor(
        input_audio, return_tensors="pt", sampling_rate=sample_rate
    ).to(device)
    with torch.no_grad():
        o = model(i.input_values)
    return o.last_hidden_state.squeeze(0)


def _get_wav2vec2_features(audio_files, output_dir, target):
    """
    ある被験者のwav2vec2の特徴量を取得する
    """
    features = []
    for audio_file in audio_files:
        wav2vec2_feature = _input_wav2vec2(audio_file)
        if wav2vec2_feature is not None:
            features.append(wav2vec2_feature)
    concatenated_features = torch.cat(features, dim=0)

    os.makedirs(os.path.join(output_dir, "wav2vec2"), exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, "wav2vec2", f"{target}.csv"),
        concatenated_features.cpu().numpy(),  # CPUに移動してからnumpyに変換する
        delimiter=",",
    )
    save_as_npy(
        os.path.join(output_dir, "wav2vec2", f"{target}.csv"),
        os.path.join(output_dir, "wav2vec2_npy"),
        skip_header=False,
    )
    return


def analyze_audio_wav2vec2(input_data_dir):
    riko_audio_dirs = os.listdir(os.path.join(input_data_dir, "voice", "riko"))
    for riko_audio_dir in riko_audio_dirs:
        if int(riko_audio_dir) < 10:
            target = f"riko0{riko_audio_dir}"
        else:
            target = f"riko{riko_audio_dir}"
        # 音声ファイルを一度に入力するとメモリが足りなくなるので、一発話ずつ入力する
        riko_audio_files = glob.glob(
            os.path.join(
                input_data_dir, "voice", "riko", riko_audio_dir, "utterance_*.wav"
            ),
            recursive=True,
        )
        riko_audio_files.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        _get_wav2vec2_features(riko_audio_files, input_data_dir, target)
    igaku_data_dirs = os.listdir(os.path.join(input_data_dir, "voice", "igaku"))
    for igaku_data_dir in igaku_data_dirs:
        # 音声ファイルを一度に入力するとメモリが足りなくなるので、一発話ずつ入力する
        igaku_audio_files = glob.glob(
            os.path.join(
                input_data_dir, "voice", "igaku", igaku_data_dir, "utterance_*.wav"
            ),
            recursive=True,
        )
        igaku_audio_files.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        _get_wav2vec2_features(
            igaku_audio_files, input_data_dir, f"psy_c_{igaku_data_dir}"
        )
    return
