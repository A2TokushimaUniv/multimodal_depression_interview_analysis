import pandas as pd
import os
import glob
from logzero import logger


def analyze_openface(before_sum_df):
    csv_file_list = glob.glob(
        os.path.join("../data/preprocessed_data/face/", "**", "*.csv"), recursive=True
    )
    for csv_file in csv_file_list:
        logger.info(f"Extracting OpenFace features from {csv_file}.")
        face_df = pd.read_csv(csv_file)
        au_intensity_columns = [
            col for col in face_df.columns if "_r" in col and "AU" in col
        ]
        au_intensity_df = face_df[au_intensity_columns]
        au_intensity_mean = au_intensity_df.mean()
        au_intensity_mean_overall = au_intensity_mean.mean()
        logger.info(f"AU_r_mean: {au_intensity_mean_overall}")
        au_intensity_std_overall = au_intensity_df.stack().std()
        logger.info(f"AU_r_std: {au_intensity_std_overall}")
        timestamp = os.path.splitext(os.path.basename(csv_file))[0]
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == timestamp, "AUr_Mean"] = (
            au_intensity_mean_overall
        )
        before_sum_df.loc[before_sum_df["タイムスタンプ"] == timestamp, "AUr_Std"] = (
            au_intensity_std_overall
        )
    return before_sum_df
