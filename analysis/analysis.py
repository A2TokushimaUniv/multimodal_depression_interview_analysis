import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    file_path = "../data/preprocessed_data/qa/before_sum_features.csv"
    df = pd.read_csv(file_path)

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

    # ヒートマップ
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
    )
    plt.title("Correlation Matrix Heatmap")
    plt.savefig("correlation_matrix_heatmap.png")


if __name__ == "__main__":
    main()
