import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils import delete_missing_ids


def plot_phq9_distribution(phq9_combined, xlabel, ylabel, output_path):
    """
    PHQ9の分布をプロットする
    """
    plt.figure(figsize=(10, 6))
    plt.hist(phq9_combined, bins=20, alpha=0.7, edgecolor="black")
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # メモリを細かく設定
    x_ticks = np.arange(0, phq9_combined.max() + 1, 1)
    plt.xticks(x_ticks)

    # 画像ファイルとして保存
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"PHQ9分布のグラフを '{output_path}' に保存しました。")
    plt.close()


def main(adult_results_path, child_results_path):
    """
    PHQの分布を計算する
    """
    # CSVデータを読み込む
    adult_results_df = pd.read_csv(adult_results_path)
    child_results_df = pd.read_csv(child_results_path)

    # 欠損データを削除
    adult_results_df = delete_missing_ids(adult_results_df)
    child_results_df = delete_missing_ids(child_results_df)

    # PHQ9列を抽出
    phq9_adult = adult_results_df["PHQ9"].dropna()
    phq9_child = child_results_df["PHQ9"].dropna()

    # 成人と子どものデータを結合
    phq9_combined = pd.concat([phq9_adult, phq9_child], ignore_index=True)
    phq9_combined.to_csv("phq9_combined.csv", index=False)

    # 平均値を計算
    adult_mean = phq9_adult.mean()
    child_mean = phq9_child.mean()
    combined_mean = phq9_combined.mean()

    print(f"成人のPHQ9平均: {adult_mean:.2f}")
    print(f"子どものPHQ9平均: {child_mean:.2f}")
    print(f"全体のPHQ9平均: {combined_mean:.2f}")

    # 日本語の図を生成
    japanize_matplotlib.japanize()
    plot_phq9_distribution(
        phq9_combined,
        xlabel="PHQ9スコア",
        ylabel="人数",
        output_path="phq9_distribution_ja.png",
    )

    # 英語の図を生成
    plt.rcParams.update({"font.size": 12})  # フォント設定をリセット
    plot_phq9_distribution(
        phq9_combined,
        xlabel="PHQ9 Score",
        ylabel="Count",
        output_path="phq9_distribution_en.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adult_results_path",
        default="../data/qa/adult_results.csv",
    )
    parser.add_argument(
        "--child_results_path",
        default="../data/qa/child_results.csv",
    )
    args = parser.parse_args()
    main(args.adult_results_path, args.child_results_path)
