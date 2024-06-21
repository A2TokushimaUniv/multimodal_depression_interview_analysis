import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logzero import logger

questionnaire_columns = {
    "Interview_Happy",
    "Interview_Anxiety",
    "Interview_Disgust",
    "Interview_Sad",
    "BIG5_Extrovert",
    "BIG5_Open",
    "BIG5_Neurotic",
    "BIG5_Diligence",
    "BIG5_Cooperative",
    "AQ",
    "PERCI",
    "GAD7",
    "LSAS",
    "PHQ9",
    "SIS",
}

multimodal_feature_columns = {
    "AUr_Mean",
    "AUr_Std",
    "Pitch",
    "Loudness",
    "Jitter",
    "HNRdBACF",
    "F0semitone",
    "F3frequency",
    "Neg_Noun_Count",
    "Neg_VerbAdj_Count",
    "Neg_Word_Count",
    "Pos_Noun_Count",
    "Pos_VerbAdj_Count",
    "Pos_Word_Count",
    "Per_Pos_Noun",
    "Per_Pos_VerbAdj",
    "Per_Neg_Noun",
    "Per_Neg_VerbAdj",
}


def main(input_file, threshold):
    df = pd.read_csv(input_file)

    logger.info("Calculating correlation matrix...")
    # タイムスタンプとLevelとFlagに関する列を削除
    columns_to_exclude = [
        col
        for col in df.columns
        if "Level" in col or "Flag" in col or "タイムスタンプ" == col
    ]
    df = df.drop(columns=columns_to_exclude)

    # 相関係数
    correlation_matrix = df.corr()
    correlation_matrix.to_csv("correlation_matrix.csv")

    # 絶対値が閾値以上のペアを見つける
    corr_pairs = correlation_matrix.unstack()
    significant_pairs = corr_pairs[abs(corr_pairs) >= threshold]
    # 同一ペアを除外
    significant_pairs = significant_pairs[
        significant_pairs.index.get_level_values(0)
        != significant_pairs.index.get_level_values(1)
    ]
    significant_pairs = significant_pairs.drop_duplicates()
    # questionnaire_columns同士のペア、multimodal_feature_columns同士のペアを削除
    significant_pairs = significant_pairs[
        ~(
            (
                significant_pairs.index.get_level_values(0).isin(questionnaire_columns)
                & significant_pairs.index.get_level_values(1).isin(
                    questionnaire_columns
                )
            )
            | (
                significant_pairs.index.get_level_values(0).isin(
                    multimodal_feature_columns
                )
                & significant_pairs.index.get_level_values(1).isin(
                    multimodal_feature_columns
                )
            )
        )
    ]

    file_suffix = str(threshold).replace(".", "")
    significant_pairs.to_csv(f"significant_pairs_{file_suffix}.csv")

    # ヒートマップ
    logger.info("Plotting heatmap...")
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
    )
    plt.title("Correlation Matrix Heatmap")
    plt.savefig("correlation_matrix_heatmap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="../data/preprocessed_data/qa/qa_result_features.csv",
        help="Input file path",
        type=str,
    )
    parser.add_argument(
        "--threshold", default=0.5, help="Correlation threshold", type=float
    )
    args = parser.parse_args()
    input_file = args.input_file
    threshold = args.threshold
    logger.info(f"Input file: {input_file}")
    logger.info(f"Threshold: {threshold}")
    main(input_file, threshold)
