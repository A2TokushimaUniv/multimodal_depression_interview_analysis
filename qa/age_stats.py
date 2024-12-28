import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils import delete_missing_ids


def plot_age_distribution(data, title, xlabel, ylabel, output_path):
    """
    年齢の分布をプロットする
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind="hist", bins=20, edgecolor="black", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 画像ファイルとして保存
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"年齢分布のグラフを '{output_path}' に保存しました。")
    plt.close()


def main(riko_before_qa_path, igaku_before_qa_path, igaku_child_before_qa_path):
    """
    平均年齢と男女比を計算する
    """
    # データフレームを読み込む
    riko_before_qa_df = pd.read_csv(riko_before_qa_path).iloc[14:]
    riko_before_qa_df = delete_missing_ids(riko_before_qa_df)
    columns_to_extract = [
        col for col in riko_before_qa_df.columns if "性別" in col or "年齢" in col
    ]
    riko_before_qa_df = riko_before_qa_df[columns_to_extract]
    riko_before_qa_df.columns = ["性別", "年齢"]

    igaku_before_qa_df = pd.read_csv(igaku_before_qa_path)
    igaku_before_qa_df = delete_missing_ids(igaku_before_qa_df)
    igaku_before_qa_df = igaku_before_qa_df[columns_to_extract]
    igaku_before_qa_df.columns = ["性別", "年齢"]

    columns_to_extract = ["年齢", "性別"]
    igaku_child_before_qa_df = pd.read_csv(igaku_child_before_qa_path)
    igaku_child_before_qa_df = delete_missing_ids(igaku_child_before_qa_df)
    igaku_child_before_qa_df = igaku_child_before_qa_df[columns_to_extract]

    # 3つのデータフレームを縦に連結
    combined_df = pd.concat(
        [riko_before_qa_df, igaku_before_qa_df, igaku_child_before_qa_df],
        ignore_index=True,
    ).dropna()

    # 年齢列の処理（数字が含まれていればそのまま、文字列なら数字部分を抽出）
    def extract_age(age):
        if isinstance(age, str):
            # 年齢が文字列の場合、数字部分を抽出
            digits = "".join(filter(str.isdigit, age))
            if digits:  # 数字部分があれば
                return int(digits)
            else:
                return np.nan  # 数字がなければNaNを返す
        elif isinstance(age, (int, float)):  # 数字がそのまま入っている場合
            return age
        return np.nan  # それ以外の場合はNaNを返す

    combined_df["年齢"] = combined_df["年齢"].apply(extract_age)
    combined_df = combined_df.dropna()  # NaNを含む行を削除

    # 男女の人数を計算
    gender_count = combined_df["性別"].value_counts()  # 性別ごとの人数をカウント
    print(f"男女の人数: \n{gender_count}")

    # 平均年齢の計算
    average_age = combined_df["年齢"].mean()
    print(f"平均年齢: {average_age:.2f}")

    # 日本語の図を生成
    japanize_matplotlib.japanize()
    plot_age_distribution(
        combined_df["年齢"],
        title="年齢の分布",
        xlabel="年齢",
        ylabel="人数",
        output_path="age_distribution_ja.png",
    )

    # 英語の図を生成
    plt.rcParams.update({"font.size": 12})  # フォント設定をリセット
    plot_age_distribution(
        combined_df["年齢"],
        title="Age Distribution",
        xlabel="Age",
        ylabel="Count",
        output_path="age_distribution_en.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--riko_before_qa_path",
        default="../data/qa/riko_before_raw.csv",
    )
    parser.add_argument(
        "--igaku_before_qa_path",
        default="../data/qa/igaku_before_raw.csv",
    )
    parser.add_argument(
        "--igaku_child_before_qa_path",
        default="../data/qa/igaku_child_before_raw.csv",
    )
    args = parser.parse_args()
    main(
        args.riko_before_qa_path,
        args.igaku_before_qa_path,
        args.igaku_child_before_qa_path,
    )
