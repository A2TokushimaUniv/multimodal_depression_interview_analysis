import opensmile
import glob
import os
from logzero import logger


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

column_names = {
    "PitchMean": "PitchMean",
    "PitchStddev": "PitchStddev",
    "LoudnessMean": "LoudnessMean",
    "LoudnessStddev": "LoudnessStddev",
    "JitterMean": "JitterMean",
    "JitterStddev": "JitterStddev",
    "ShimmerMean": "ShimmerMean",
    "ShimmerStddev": "ShimmerStddev",
    "HNRdBACF": "HNRdBACF",
    "F0semitone": "F0semitone",
    "F3frequency": "F3frequency",
}


def get_pitch_features(features):
    pitch_mean = features["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[-1]
    logger.info(f"pitch_mean: {pitch_mean}")
    pitch_stddev = features["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"pitch_stddev: {pitch_stddev}")
    return pitch_mean, pitch_stddev


def get_loudness_features(features):
    loudness_mean = features["loudness_sma3_amean"].iloc[-1]
    logger.info(f"loudness_mean: {loudness_mean}")
    loudness_stddev = features["loudness_sma3_stddevNorm"].iloc[-1]
    logger.info(f"loudness_stddev: {loudness_stddev}")
    return loudness_mean, loudness_stddev


def get_jitter_features(features):
    jitter_mean = features["jitterLocal_sma3nz_amean"].iloc[-1]
    logger.info(f"jitter_mean: {jitter_mean}")
    jitter_stddev = features["jitterLocal_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"jitter_stddev: {jitter_stddev}")
    return jitter_mean, jitter_stddev


def get_shimmer_features(features):
    shimmer_mean = features["shimmerLocal_sma3nz_amean"].iloc[-1]
    logger.info(f"shimmer_mean: {shimmer_mean}")
    shimmer_stddev = features["shimmerLocal_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"shimmer_stddev: {shimmer_stddev}")
    return shimmer_mean, shimmer_stddev


def get_other_features(features):
    HNRdBACF_sma3nz = features["HNRdBACF_sma3nz"].iloc[-1]
    logger.info(f"HNRdBACF_sma3nz: {HNRdBACF_sma3nz}")
    F0semitone = features["F0semitoneFrom27.5Hz_sma3nz"].iloc[-1]
    logger.info(f"F0semitone: {F0semitone}")
    F3frequency = features["F3frequency_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"F3frequency: {F3frequency}")
    return HNRdBACF_sma3nz, F0semitone, F3frequency


def add_results(
    qa_result_df,
    target,
    pitch_mean,
    pitch_stddev,
    loudness_mean,
    loudness_stddev,
    jitter_mean,
    jitter_stddev,
    shimmer_mean,
    shimmer_stddev,
    HNRdBACF_sma3nz,
    F0semitone,
    F3frequency,
):
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["PitchMean"]
    ] = pitch_mean
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["PitchStddev"]
    ] = pitch_stddev
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["LoudnessMean"]
    ] = loudness_mean
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["LoudnessStddev"]
    ] = loudness_stddev
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["JitterMean"]
    ] = jitter_mean
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["JitterStddev"]
    ] = jitter_stddev
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["ShimmerMean"]
    ] = shimmer_mean
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["ShimmerStddev"]
    ] = shimmer_stddev
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["HNRdBACF"]
    ] = HNRdBACF_sma3nz
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["F0semitone"]
    ] = F0semitone
    qa_result_df.loc[
        qa_result_df["タイムスタンプ"] == target, column_names["F3frequency"]
    ] = F3frequency
    return qa_result_df


def analyze_audio(qa_result_df):
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
        features = smile.process_file(riko_audio_file)
        pitch_mean, pitch_stddev = get_pitch_features(features)
        loudness_mean, loudness_stddev = get_loudness_features(features)
        jitter_mean, jitter_stddev = get_jitter_features(features)
        shimmer_mean, shimmer_stddev = get_shimmer_features(features)
        HNRdBACF_sma3nz, F0semitone, F3frequency = get_other_features(features)
        id = riko_audio_file.split("/")[-2]
        if int(id) < 10:
            target = f"riko0{id}"
        else:
            target = f"riko{id}"

        qa_result_df = add_results(
            qa_result_df,
            target,
            pitch_mean,
            pitch_stddev,
            loudness_mean,
            loudness_stddev,
            jitter_mean,
            jitter_stddev,
            shimmer_mean,
            shimmer_stddev,
            HNRdBACF_sma3nz,
            F0semitone,
            F3frequency,
        )

    for igaku_audio_file in igaku_audio_files:
        logger.info(f"Extracting openSMILE features from {igaku_audio_file}....")
        features = smile.process_file(igaku_audio_file)
        pitch_mean, pitch_stddev = get_pitch_features(features)
        loudness_mean, loudness_stddev = get_loudness_features(features)
        jitter_mean, jitter_stddev = get_jitter_features(features)
        shimmer_mean, shimmer_stddev = get_shimmer_features(features)
        HNRdBACF_sma3nz, F0semitone, F3frequency = get_other_features(features)
        id = igaku_audio_file.split("/")[-2]
        target = f"psy_c_{id}"
        qa_result_df = add_results(
            qa_result_df,
            target,
            pitch_mean,
            pitch_stddev,
            loudness_mean,
            loudness_stddev,
            jitter_mean,
            jitter_stddev,
            shimmer_mean,
            shimmer_stddev,
            HNRdBACF_sma3nz,
            F0semitone,
            F3frequency,
        )

    return qa_result_df
