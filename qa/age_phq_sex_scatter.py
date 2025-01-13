import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 目盛り設定用に追加
import japanize_matplotlib


def main(age_sex_path, phq9_path):
    # データの読み込み
    age_sex_df = pd.read_csv(age_sex_path)
    phq9_df = pd.read_csv(phq9_path)

    # データの結合
    all_df = pd.concat([age_sex_df, phq9_df], axis=1)
    all_df.to_csv("all_combined.csv", index=False)

    # 散布図の作成と保存
    plot_scatter(
        all_df,
        output_path_japanese="scatter_plot_jp.png",
        output_path_english="scatter_plot_en.png",
    )
    return


def plot_scatter(
    all_df,
    output_path_japanese="scatter_plot_jp.png",
    output_path_english="scatter_plot_en.png",
):
    # 日本語版の散布図
    plt.figure(figsize=(10, 6))

    # 男性のデータ
    male_df = all_df[all_df["性別"] == "男性"]
    plt.scatter(male_df["PHQ9"], male_df["年齢"], color="blue", label="男性", alpha=0.7)

    # 女性のデータ
    female_df = all_df[all_df["性別"] == "女性"]
    plt.scatter(
        female_df["PHQ9"], female_df["年齢"], color="red", label="女性", alpha=0.7
    )

    # グラフの装飾
    japanize_matplotlib.japanize()
    plt.xlabel("PHQ9", fontsize=12)
    plt.ylabel("年齢", fontsize=12)
    plt.legend()
    plt.grid(True)

    # 横軸を5刻みで区切る
    x_min, x_max = int(all_df["PHQ9"].min()), int(all_df["PHQ9"].max())
    x_ticks = np.arange(0, x_max + 5, 5)  # 0から最大値を超える範囲で5刻み
    plt.xticks(x_ticks)

    # グラフの保存
    plt.savefig(output_path_japanese, format="png", dpi=300)
    print(f"日本語版の散布図を {output_path_japanese} に保存しました。")
    plt.close()

    # 英語版の散布図
    plt.figure(figsize=(10, 6))

    # 男性のデータ
    plt.scatter(male_df["PHQ9"], male_df["年齢"], color="blue", label="Male", alpha=0.7)

    # 女性のデータ
    plt.scatter(
        female_df["PHQ9"], female_df["年齢"], color="red", label="Female", alpha=0.7
    )

    # グラフの装飾
    plt.xlabel("PHQ9", fontsize=12)
    plt.ylabel("Age", fontsize=12)
    plt.legend()
    plt.grid(True)

    # 横軸を5刻みで区切る
    plt.xticks(x_ticks)

    # グラフの保存
    plt.savefig(output_path_english, format="png", dpi=300)
    print(f"English version of scatter plot saved to {output_path_english}.")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--age_sex_path",
        default="age_sex_combined.csv",
    )
    parser.add_argument(
        "--phq9_path",
        default="phq9_combined.csv",
    )
    args = parser.parse_args()
    main(
        args.age_sex_path,
        args.phq9_path,
    )
