import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from logzero import logger


def get_significant_pairs(output_dir, correlation_matrix, threshold):
    """
    相関係数の絶対値が閾値以上のペアを見つける
    """
    corr_pairs = correlation_matrix.unstack()
    significant_pairs = corr_pairs[abs(corr_pairs) >= threshold]
    significant_pairs = significant_pairs[
        significant_pairs.index.get_level_values(0)
        != significant_pairs.index.get_level_values(1)
    ]
    significant_pairs = significant_pairs.drop_duplicates()

    # p値を計算し、辞書に格納
    p_values = {}
    for pair in significant_pairs.index:
        col1, col2 = pair
        _, p_value = stats.pearsonr(correlation_matrix[col1], correlation_matrix[col2])
        p_values[pair] = p_value

    # 結果をデータフレームに変換
    significant_pairs_df = pd.DataFrame(significant_pairs, columns=["correlation"])
    significant_pairs_df["p_value"] = significant_pairs_df.index.map(p_values)

    # インデックスをカラムに変換
    significant_pairs_df.reset_index(inplace=True)
    significant_pairs_df.columns = ["Feature1", "Feature2", "correlation", "p_value"]

    # questionnaire_columns同士のペア、multimodal_feature_columns同士のペアを削除
    questionnaire_columns = (
        "Interview_Happy",
        "Interview_Anxiety",
        "Interview_Disgust",
        "Interview_Sad",
        "BIG5_Extrovert",
        "BIG5_Openness",
        "BIG5_Neuroticism",
        "BIG5_Conscientiousness",
        "BIG5_Agreeableness",
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
        "CharPerMinutes",
        "WordPerMinutes",
    ]

    multimodal_feature_columns = set(au_columns + feature_columns)
    significant_pairs_df = significant_pairs_df[
        ~(
            (
                significant_pairs_df["Feature1"].isin(questionnaire_columns)
                & significant_pairs_df["Feature2"].isin(questionnaire_columns)
            )
            | (
                significant_pairs_df["Feature1"].isin(multimodal_feature_columns)
                & significant_pairs_df["Feature2"].isin(multimodal_feature_columns)
            )
        )
    ]

    file_suffix = str(threshold).replace(".", "")
    significant_pairs_df.to_csv(
        os.path.join(output_dir, f"significant_pairs_{file_suffix}.csv"), index=False
    )


def get_heatmap(output_dir, correlation_matrix):
    """
    相関行列のヒートマップをプロットする
    """
    logger.info("Plotting heatmap...")
    plt.figure(figsize=(50, 30))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
    )
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_matrix_heatmap.png"))


def calculate_statistics(output_dir, df):
    """
    特徴量の平均と標準偏差を計算する
    """
    means = df.mean()
    stds = df.std()

    results = pd.DataFrame({"Mean": means, "Standard Deviation": stds})

    results.to_csv(os.path.join(output_dir, "column_statistics.csv"), index=True)


def main(input_file_path, output_dir, threshold):
    df = pd.read_csv(input_file_path)

    logger.info("Calculating correlation matrix...")
    columns_to_exclude = [
        col for col in df.columns if "Level" in col or "Flag" in col or "ID" == col
    ]
    df = df.drop(columns=columns_to_exclude)
    calculate_statistics(output_dir, df)
    correlation_matrix = df.corr()
    correlation_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    get_significant_pairs(output_dir, correlation_matrix, threshold)
    get_heatmap(output_dir, correlation_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        default="../data/qa/qa_result_features.csv",
        help="Input file path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="./analysis_results",
        help="Output directory",
        type=str,
    )
    parser.add_argument(
        "--threshold", default=0.3, help="Correlation threshold", type=float
    )
    args = parser.parse_args()
    input_file_path = args.input_file_path
    output_dir = args.output_dir
    threshold = args.threshold
    logger.info(f"Input file: {input_file_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Threshold: {threshold}")
    main(input_file_path, output_dir, threshold)
