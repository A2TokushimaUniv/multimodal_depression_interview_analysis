import opensmile
import glob
import os
from logzero import logger
from utils import save_as_npy

"""
OpenSMILEの設定、特徴量セットにはeGeMAPSv02を使用
"""
smile_functions = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,  # LLDの統計値を計算する
)

smile_llds = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,  # LLDを計算する
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
    """
    ピッチ関連の特徴量を取得する
    """
    pitch_mean = features["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[-1]
    logger.info(f"PitchMean: {pitch_mean}")
    pitch_stddev = features["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"PitchStddev: {pitch_stddev}")
    return pitch_mean, pitch_stddev


def get_loudness_features(features):
    """
    声の大きさ関連の特徴量を取得
    """
    loudness_mean = features["loudness_sma3_amean"].iloc[-1]
    logger.info(f"LoudnessMean: {loudness_mean}")
    loudness_stddev = features["loudness_sma3_stddevNorm"].iloc[-1]
    logger.info(f"LoudnessStddev: {loudness_stddev}")
    return loudness_mean, loudness_stddev


def get_jitter_features(features):
    """
    ジッター関連の特徴量を取得
    """
    jitter_mean = features["jitterLocal_sma3nz_amean"].iloc[-1]
    logger.info(f"JitterMean: {jitter_mean}")
    jitter_stddev = features["jitterLocal_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"JitterStddev: {jitter_stddev}")
    return jitter_mean, jitter_stddev


def get_shimmer_features(features):
    """
    shimmerr関連の特徴量を取得
    """
    shimmer_mean = features["shimmerLocaldB_sma3nz_amean"].iloc[-1]
    logger.info(f"ShimmerMean: {shimmer_mean}")
    shimmer_stddev = features["shimmerLocaldB_sma3nz_stddevNorm"].iloc[-1]
    logger.info(f"ShimmerStddev: {shimmer_stddev}")
    return shimmer_mean, shimmer_stddev


def get_other_features(features):
    """
    その他の特徴量を取得
    """
    HNRdBACF_sma3nz = features["HNRdBACF_sma3nz_amean"].iloc[-1]
    logger.info(f"HNRdBACF_sma3nz: {HNRdBACF_sma3nz}")
    F0semitone = features["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"].iloc[-1]
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
    """
    結果をDataFrameに追加する
    """
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


def save_llds(llds_features, output_dir, target):
    """
    LLDsを保存する
    """
    os.makedirs(os.path.join(output_dir, "opensmile"), exist_ok=True)
    csv_file_path = os.path.join(output_dir, "opensmile", f"{target}.csv")
    llds_features.to_csv(csv_file_path, index=False)
    save_as_npy(csv_file_path, os.path.join(output_dir, "opensmile_npy"))


def analyze_audio_opensmile(qa_result_df, input_data_dir):
    """
    音声をOpenSMILEで特徴量抽出する
    """
    riko_audio_files = glob.glob(
        os.path.join(input_data_dir, "voice", "riko", "*", "audioNLP*.wav"),
        recursive=True,
    )
    igaku_audio_files = glob.glob(
        os.path.join(input_data_dir, "voice", "igaku", "*", "*_zoom_音声_被験者*.wav"),
        recursive=True,
    )
    for riko_audio_file in riko_audio_files:
        logger.info(f"Extracting openSMILE stats features from {riko_audio_file}....")
        stats_features = smile_functions.process_file(riko_audio_file)
        pitch_mean, pitch_stddev = get_pitch_features(stats_features)
        loudness_mean, loudness_stddev = get_loudness_features(stats_features)
        jitter_mean, jitter_stddev = get_jitter_features(stats_features)
        shimmer_mean, shimmer_stddev = get_shimmer_features(stats_features)
        HNRdBACF_sma3nz, F0semitone, F3frequency = get_other_features(stats_features)

        logger.info(f"Extracting openSMILE LLDs features from {riko_audio_file}....")
        data_id = riko_audio_file.split("/")[-2]
        if int(data_id) < 10:
            target = f"riko0{data_id}"
        else:
            target = f"riko{data_id}"
        llds_features = smile_llds.process_file(riko_audio_file)
        save_llds(llds_features, input_data_dir, target)

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
        logger.info(f"Extracting openSMILE stats features from {igaku_audio_file}....")
        stats_features = smile_functions.process_file(igaku_audio_file)
        pitch_mean, pitch_stddev = get_pitch_features(stats_features)
        loudness_mean, loudness_stddev = get_loudness_features(stats_features)
        jitter_mean, jitter_stddev = get_jitter_features(stats_features)
        shimmer_mean, shimmer_stddev = get_shimmer_features(stats_features)
        HNRdBACF_sma3nz, F0semitone, F3frequency = get_other_features(stats_features)

        logger.info(f"Extracting openSMILE LLDs features from {igaku_audio_file}....")
        data_id = igaku_audio_file.split("/")[-2]
        target = f"psy_c_{data_id}"
        llds_features = smile_llds.process_file(igaku_audio_file)
        save_llds(llds_features, input_data_dir, target)

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
