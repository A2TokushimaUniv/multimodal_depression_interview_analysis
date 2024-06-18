import opensmile
import glob
import os
from logzero import logger


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

column_names = {
    "Pitch": "Pitch",
    "Loudness": "Loudness",
    "Jitter": "Jitter",
    "HNRdBACF": "HNRdBACF",
    "F0semitone": "F0semitone",
    "F3frequency": "F3frequency",
}


def get_features(audio_file):
    features = smile.process_file(audio_file)
    pitch = features["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[-1]
    logger.info(f"pitch: {pitch}")
    loudness = features["loudness_sma3_amean"].iloc[-1]
    logger.info(f"loudness: {loudness}")
    jitter = features["jitterLocal_sma3nz_amean"].iloc[-1]
    logger.info(f"jitter: {jitter}")
    HNRdBACF_sma3nz = features["HNRdBACF_sma3nz_amean"].iloc[-1]
    logger.info(f"HNRdBACF: {HNRdBACF_sma3nz}")
    F0semitone = features["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"].iloc[-1]
    logger.info(f"F0semitone: {F0semitone}")
    F3frequency = features["F3frequency_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"F3frequency: {F3frequency}")
    return pitch, loudness, jitter, HNRdBACF_sma3nz, F0semitone, F3frequency


def add_results(
    before_sum_df,
    target,
    pitch,
    loudness,
    jitter,
    HNRdBACF_sma3nz,
    F0semitone,
    F3frequency,
):
    before_sum_df.loc[
        before_sum_df["タイムスタンプ"] == target, column_names["Pitch"]
    ] = pitch
    before_sum_df.loc[
        before_sum_df["タイムスタンプ"] == target, column_names["Loudness"]
    ] = loudness
    before_sum_df.loc[
        before_sum_df["タイムスタンプ"] == target, column_names["Jitter"]
    ] = jitter
    before_sum_df.loc[
        before_sum_df["タイムスタンプ"] == target, column_names["HNRdBACF"]
    ] = HNRdBACF_sma3nz
    before_sum_df.loc[
        before_sum_df["タイムスタンプ"] == target, column_names["F0semitone"]
    ] = F0semitone
    before_sum_df.loc[
        before_sum_df["タイムスタンプ"] == target, column_names["F3frequency"]
    ] = F3frequency
    return before_sum_df


def analyze_opensmile(before_sum_df):
    riko_audio_files = glob.glob(
        os.path.join("../data/preprocessed_data/voice/riko", "*", "audioNLP*.wav"),
        recursive=True,
    )
    igaku_audio_files = glob.glob(
        os.path.join(
            "../data/preprocessed_data/voice/igaku", "*", "*_zoom_音声_被験者*.wav"
        ),
        recursive=True,
    )

    for riko_audio_file in riko_audio_files:
        logger.info(f"Extracting openSMILE features from {riko_audio_file}....")
        pitch, loudness, jitter, HNRdBACF_sma3nz, F0semitone, F3frequency = (
            get_features(riko_audio_file)
        )
        id = riko_audio_file.split("/")[-2]
        if int(id) < 10:
            target = f"riko0{id}"
        else:
            target = f"riko{id}"

        before_sum_df = add_results(
            before_sum_df,
            target,
            pitch,
            loudness,
            jitter,
            HNRdBACF_sma3nz,
            F0semitone,
            F3frequency,
        )

    for igaku_audio_file in igaku_audio_files:
        logger.info(f"Extracting openSMILE features from {igaku_audio_file}....")
        pitch, loudness, jitter, HNRdBACF_sma3nz, F0semitone, F3frequency = (
            get_features(igaku_audio_file)
        )
        id = igaku_audio_file.split("/")[-2]
        before_sum_df = add_results(
            before_sum_df,
            target,
            pitch,
            loudness,
            jitter,
            HNRdBACF_sma3nz,
            F0semitone,
            F3frequency,
        )

    return before_sum_df
