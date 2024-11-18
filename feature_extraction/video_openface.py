import pandas as pd
import os
from logzero import logger
from utils import get_openface_files


def _get_results(csv_file, qa_result_df, data_id):
    """
    結果をDataFrameに追加する
    """
    logger.info(f"{csv_file}からOpenFace特徴量の統計値を計算しています....")
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
    # AU全体の標準偏差を計算
    au_intensity_std_overall = au_intensity_df.stack().std()

    qa_result_df.loc[qa_result_df["ID"] == data_id, "AUall_r_Mean"] = (
        au_intensity_mean_overall
    )
    qa_result_df.loc[qa_result_df["ID"] == data_id, "AUall_r_Stddev"] = (
        au_intensity_std_overall
    )
    for au in au_intensity_columns:
        qa_result_df.loc[qa_result_df["ID"] == data_id, f"{au}_Mean".strip()] = (
            au_intensity_mean[au]
        )
        qa_result_df.loc[qa_result_df["ID"] == data_id, f"{au}_Stddev".strip()] = (
            au_intensity_std[au]
        )

    return qa_result_df


def analyze_openface_stats(adult_qa_df, child_qa_df, input_data_dir):
    """
    OpenFaceの特徴量を使って統計値を計算する
    """
    openface_files = get_openface_files(os.path.join(input_data_dir, "openface"))
    for data_id, openface_file in openface_files:
        adult_qa_df = _get_results(openface_file, adult_qa_df, data_id)
        child_qa_df = _get_results(openface_file, child_qa_df, data_id)
    return adult_qa_df, child_qa_df
