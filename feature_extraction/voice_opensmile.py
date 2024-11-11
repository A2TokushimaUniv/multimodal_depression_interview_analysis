import opensmile
import os
from logzero import logger
from utils import save_feature, get_voice_files
import pandas as pd
import librosa

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


def _get_pitch(feature):
    """
    ピッチ関連の特徴量を取得する
    """
    pitch_mean = feature["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[-1]
    pitch_stddev = feature["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"].iloc[-1]
    return pitch_mean, pitch_stddev


def _get_loudness(feature):
    """
    声の大きさ関連の特徴量を取得
    """
    loudness_mean = feature["loudness_sma3_amean"].iloc[-1]
    loudness_stddev = feature["loudness_sma3_stddevNorm"].iloc[-1]
    return loudness_mean, loudness_stddev


def _get_jitter(feature):
    """
    ジッター関連の特徴量を取得
    """
    jitter_mean = feature["jitterLocal_sma3nz_amean"].iloc[-1]
    jitter_stddev = feature["jitterLocal_sma3nz_stddevNorm"].iloc[-1]
    return jitter_mean, jitter_stddev


def _get_shimmer(feature):
    """
    shimmerr関連の特徴量を取得
    """
    shimmer_mean = feature["shimmerLocaldB_sma3nz_amean"].iloc[-1]
    shimmer_stddev = feature["shimmerLocaldB_sma3nz_stddevNorm"].iloc[-1]
    return shimmer_mean, shimmer_stddev


def _get_others(feature):
    """
    その他の特徴量を取得
    """
    HNRdBACF_sma3nz = feature["HNRdBACF_sma3nz_amean"].iloc[-1]
    F0semitone = feature["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"].iloc[-1]
    F3frequency = feature["F3frequency_sma3nz_stddevNorm"].iloc[-1]
    return HNRdBACF_sma3nz, F0semitone, F3frequency


def _add_results(
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
        qa_result_df["Subject_ID"] == target, column_names["PitchMean"]
    ] = pitch_mean
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["PitchStddev"]
    ] = pitch_stddev
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["LoudnessMean"]
    ] = loudness_mean
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["LoudnessStddev"]
    ] = loudness_stddev
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["JitterMean"]
    ] = jitter_mean
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["JitterStddev"]
    ] = jitter_stddev
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["ShimmerMean"]
    ] = shimmer_mean
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["ShimmerStddev"]
    ] = shimmer_stddev
    qa_result_df.loc[qa_result_df["Subject_ID"] == target, column_names["HNRdBACF"]] = (
        HNRdBACF_sma3nz
    )
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["F0semitone"]
    ] = F0semitone
    qa_result_df.loc[
        qa_result_df["Subject_ID"] == target, column_names["F3frequency"]
    ] = F3frequency
    return qa_result_df


def analyze_opensmile_stats(qa_result_df, input_data_dir):
    """
    音声からopenSMILEの特徴量の統計値を取得し、その平均と標準偏差を計算する
    """
    voice_files = get_voice_files(input_data_dir)

    for data_id, voice_file in voice_files:
        logger.info(f"Extracting openSMILE stats feature from {voice_file}....")
        stats_feature = smile_functions.process_file(voice_file)
        pitch_mean, pitch_stddev = _get_pitch(stats_feature)
        loudness_mean, loudness_stddev = _get_loudness(stats_feature)
        jitter_mean, jitter_stddev = _get_jitter(stats_feature)
        shimmer_mean, shimmer_stddev = _get_shimmer(stats_feature)
        HNRdBACF_sma3nz, F0semitone, F3frequency = _get_others(stats_feature)

        qa_result_df = _add_results(
            qa_result_df,
            data_id,
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


def _get_lld(voice_path):
    """
    D-Vlogの元論文に記載されている方法でLLDを抽出する
    1秒毎にLLDsを抽出・平均化し、全てのLLDsを連結したものを特徴量とする
    frameStep, frameSizeの合わせ方は不明なので無視する
    """
    y, sr = librosa.load(voice_path, sr=None)
    duration = int(librosa.get_duration(y=y, sr=sr))

    lld_data = []
    # 1秒ごとに音声を処理する
    for i in range(duration):
        start_sample = int(i * sr)
        end_sample = int((i + 1) * sr)
        y_segment = y[start_sample:end_sample]
        # 音声セグメントが1秒に満たない場合はループを終了
        if len(y_segment) < sr:
            break
        lld_result = smile_llds.process_signal(y_segment, sampling_rate=sr)
        lld_mean = lld_result.mean(axis=0)
        lld_data.append(lld_mean)
    lld_df = pd.DataFrame(lld_data)
    if duration != lld_df.shape[0]:
        logger.warning(f"duration: {duration} != lld feature column: {lld_df.shape[0]}")
    return lld_df


def extract_opensmile_lld_feature(input_data_dir, output_data_dir):
    """
    音声からopenSMILEのLLD特徴量を抽出する
    """
    voice_files = get_voice_files(input_data_dir)
    for data_id, voice_file in voice_files:
        logger.info(f"Extracting openSMILE LLD feature from {voice_file}....")
        feature = _get_lld(voice_file)
        save_feature(
            feature,
            os.path.join(output_data_dir, "opensmile"),
            f"{data_id}.csv",
        )
