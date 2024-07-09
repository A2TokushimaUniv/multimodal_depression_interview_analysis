import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logzero import logger


def get_significant_pairs(correlation_matrix, threshold):
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
    questionnaire_columns = (
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
    )

    openface_au_list = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
    au_columns = [
        f"AU{'0' if au < 10 else ''}{au}_r_{stat}".strip()
        for au in openface_au_list
        for stat in ["Mean", "Stddev"]
    ]
    feature_columns = [
        "AUall_r_Mean",
        "AUall_r_Stddev",
        "PitchMean",
        "PitchStddev",
        "LoudnessMean",
        "LoudnessStddev",
        "JitterMean",
        "JitterStddev",
        "ShimmerMean",
        "ShimmerStddev",
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
    ]

    multimodal_feature_columns = set(au_columns + feature_columns)
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


def get_heatmap(correlation_matrix):
    logger.info("Plotting heatmap...")
    plt.figure(figsize=(50, 30))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
    )
    plt.title("Correlation Matrix Heatmap")
    plt.savefig("correlation_matrix_heatmap.png")


def calculate_statistics(df):
    means = df.mean()
    stds = df.std()

    results = pd.DataFrame({"Mean": means, "Standard Deviation": stds})

    output_csv_file_path = "column_statistics.csv"
    results.to_csv(output_csv_file_path, index=True)


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
    calculate_statistics(df)
    correlation_matrix = df.corr()
    correlation_matrix.to_csv("correlation_matrix.csv")
    get_significant_pairs(correlation_matrix, threshold)
    get_heatmap(correlation_matrix)


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
