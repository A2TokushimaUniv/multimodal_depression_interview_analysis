import pandas as pd
import argparse
import numpy as np
from utils import delete_missing_ids


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
