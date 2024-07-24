import pandas as pd
import os
import glob
from logzero import logger


def analyze_face(qa_result_df, input_data_dir):
    riko_csv_file_list = glob.glob(
        os.path.join(input_data_dir, "face", "riko", "*.csv"), recursive=True
    )
    igaku_csv_file_list = glob.glob(
        os.path.join(input_data_dir, "face", "igaku", "*.csv"), recursive=True
    )
    csv_file_list = riko_csv_file_list + igaku_csv_file_list
    for csv_file in csv_file_list:
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

        timestamp = os.path.splitext(os.path.basename(csv_file))[0]
        qa_result_df.loc[
            qa_result_df["タイムスタンプ"] == timestamp, "AUall_r_Mean"
        ] = au_intensity_mean_overall
        qa_result_df.loc[
            qa_result_df["タイムスタンプ"] == timestamp, "AUall_r_Stddev"
        ] = au_intensity_std_overall
        for au in au_intensity_columns:
            qa_result_df.loc[
                qa_result_df["タイムスタンプ"] == timestamp, f"{au}_Mean".strip()
            ] = au_intensity_mean[au]
            logger.info(f"{au}_Mean: {au_intensity_mean[au]}")
            qa_result_df.loc[
                qa_result_df["タイムスタンプ"] == timestamp, f"{au}_Stddev".strip()
            ] = au_intensity_std[au]
            logger.info(f"{au}_Stddev: {au_intensity_std[au]}")

    return qa_result_df
