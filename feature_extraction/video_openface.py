import pandas as pd
import os
import glob
from logzero import logger
from utils import save_as_npy


def _get_results(csv_file, qa_result_df):
    """
    結果をDataFrameに追加する
    """
    logger.info(f"Extracting OpenFace features from {csv_file}....")
    face_df = pd.read_csv(csv_file)
    au_intensity_columns = [
        col for col in face_df.columns if "_r" in col and "AU" in col
    ]
    au_intensity_df = face_df[au_intensity_columns]
    # AUごとの平均値と標準偏差を計算
    au_intensity_mean = au_intensity_df.mean()
    au_intensity_std = au_intensity_df.std()
    # AU全体の平均を計算
    au_intensity_mean_overall = au_intensity_mean.mean()
    logger.info(f"AU_r_Mean: {au_intensity_mean_overall}")
    # AU全体の標準偏差を計算
    au_intensity_std_overall = au_intensity_df.stack().std()
    logger.info(f"AU_r_Stddev: {au_intensity_std_overall}")

    subject_id = os.path.splitext(os.path.basename(csv_file))[0]
    qa_result_df.loc[qa_result_df["Subject_ID"] == subject_id, "AUall_r_Mean"] = (
        au_intensity_mean_overall
    )
    qa_result_df.loc[qa_result_df["Subject_ID"] == subject_id, "AUall_r_Stddev"] = (
        au_intensity_std_overall
    )
    for au in au_intensity_columns:
        qa_result_df.loc[
            qa_result_df["Subject_ID"] == subject_id, f"{au}_Mean".strip()
        ] = au_intensity_mean[au]
        logger.info(f"{au}_Mean: {au_intensity_mean[au]}")
        qa_result_df.loc[
            qa_result_df["Subject_ID"] == subject_id, f"{au}_Stddev".strip()
        ] = au_intensity_std[au]
        logger.info(f"{au}_Stddev: {au_intensity_std[au]}")

    return qa_result_df


def analyze_openface_stats(qa_result_df, input_data_dir):
    """
    OpenFaceの特徴量を使って統計量を計算する
    """
    openface_dir = "openface"
    csv_files = glob.glob(
        os.path.join(input_data_dir, openface_dir, "*.csv"), recursive=True
    )

    # TODO: LMVDとDAIC-WOZと列を合わせたものを生成する
    for csv_file in csv_files:
        save_as_npy(csv_file, os.path.join(input_data_dir, f"{openface_dir}_npy"))
        qa_result_df = _get_results(csv_file, qa_result_df)
    return qa_result_df
