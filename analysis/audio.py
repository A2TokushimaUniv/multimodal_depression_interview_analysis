import opensmile
import glob
import os
from logzero import logger


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def calculate_features(audio_file):
    features = smile.process_file(audio_file)
    pitch = features["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[-1]
    loudness = features["loudness_sma3_amean"].iloc[-1]
    jitter = features["jitterLocal_sma3nz_amean"].iloc[-1]
    return pitch, loudness, jitter


def analyze_opensmile(before_sum_df):
    riko_audio_files = glob.glob(
        os.path.join("../data/preprocessed_data/audio/riko", "**", "audioNLP*.wav"),
        recursive=True,
    )
    igaku_audio_files = glob.glob(
        os.path.join(
            "../data/preprocessed_data/video/igaku", "**", "*_zoom_音声_被験者*.wav"
        ),
        recursive=True,
    )

    for riko_audio_file in riko_audio_files:
        logger.info(f"Extracting openSMILE features from {riko_audio_file}.")
        pitch, loudness, jitter = calculate_features(riko_audio_file)
        id = riko_audio_file.split("/")[-2]
        if int(id) < 10:
            target = f"riko0{id}"
        else:
            target = f"riko{id}"

        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "Pitch"] = pitch
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "Loudness"] = (
            loudness
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == target, "Jitter"] = jitter

    for igaku_audio_file in igaku_audio_files:
        logger.info(f"Extracting openSMILE features from {igaku_audio_file}.")
        pitch, loudness, jitter = calculate_features(igaku_audio_file)
        id = igaku_audio_file.split("/")[-2]
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == f"psy_c_{id}", "Pitch"] = (
            pitch
        )
        before_sum_df.loc[
            before_sum_df["タイムスタンプ"] == f"psy_c_{id}", "Loudness"
        ] = loudness
        before_sum_df.loc[
            before_sum_df["タイムスタンプ"] == f"psy_c_{id}", "Jitter"
        ] = jitter
    return before_sum_df
