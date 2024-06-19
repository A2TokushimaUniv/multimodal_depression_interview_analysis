import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logzero import logger


def main(threshold):
    file_path = "../data/preprocessed_data/qa/qa_result_features.csv"
    df = pd.read_csv(file_path)

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
        "--threshold", default=0.5, help="Correlation threshold", type=float
    )
    args = parser.parse_args()
    threshold = args.threshold
    logger.info(f"Threshold: {threshold}")
    main(threshold)
