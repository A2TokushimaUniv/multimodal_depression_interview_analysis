import argparse
from logzero import logger
from utils import is_csv, add_riko_id, extract_columns
import pandas as pd
from before import convert_adult, convert_child


def main(
    riko_before_file_path,
    igaku_before_file_path,
    igaku_child_before_file_path,
    igaku_parent_before_file_path,
    riko_after_file_path,
    igaku_after_file_path,
):
    if not is_csv(
        [
            riko_before_file_path,
            igaku_before_file_path,
            igaku_child_before_file_path,
            igaku_parent_before_file_path,
            riko_after_file_path,
            igaku_after_file_path,
        ]
    ):
        logger.error("CSV以外のファイルが含まれています")
        raise ValueError("CSV以外のファイルが含まれています")

    before_columns = pd.read_csv("./columns/before_adult.csv")
    before_child_parent_columns = pd.read_csv("./columns/before_child_parent.csv")
    after_columns = pd.read_csv("./columns/after.csv")

    # 成人のアンケートの処理
    riko_before_df = pd.read_csv(riko_before_file_path)
    riko_before_df["タイムスタンプ"] = pd.to_datetime(riko_before_df["タイムスタンプ"])
    riko_before_df = add_riko_id(
        riko_before_df.sort_values(by="タイムスタンプ", ascending=True).iloc[14:]
    )  # 最初の14行はテスト時のデータなのでスキップ
    extracted_riko_before_df = extract_columns(
        riko_before_df, before_columns
    )  # 必要な列を抽出
    igaku_before_df = pd.read_csv(igaku_before_file_path)
    extracted_igaku_before_df = extract_columns(
        igaku_before_df, before_columns
    )  # 必要な列を抽出
    before_df = pd.concat(
        [extracted_riko_before_df, extracted_igaku_before_df], axis=0
    )  # 事前アンケートを縦方向に連結
    converted_before_df = convert_adult(before_df)  # 各質問項目を数値に変換する

    riko_after_df = pd.read_csv(riko_after_file_path)
    riko_after_df["タイムスタンプ"] = pd.to_datetime(riko_after_df["タイムスタンプ"])
    riko_after_df = add_riko_id(
        riko_after_df.sort_values(by="タイムスタンプ", ascending=True).iloc[3:]
    )  # 最初の3行はテスト時のデータなのでスキップ
    extracted_riko_after_df = extract_columns(
        riko_after_df, after_columns
    )  # 必要な列を抽出
    igaku_after_df = pd.read_csv(igaku_after_file_path)
    extracted_igaku_after_df = extract_columns(
        igaku_after_df, after_columns
    )  # 必要な列を抽出
    extracted_igaku_adult_after_df = extracted_igaku_after_df[
        extracted_igaku_after_df["ID"].isin(igaku_before_df["ID"])
    ]  # 成人のデータのみを取り出す
    after_df = pd.concat(
        [extracted_riko_after_df, extracted_igaku_adult_after_df], axis=0
    )  # 事後アンケートを縦方向に連結
    after_df.columns = [
        "ID",
        "Interview_Happy",
        "Interview_Anxiety",
        "Interview_Disgust",
        "Interview_Sad",
    ]
    qa_adult_results_df = pd.merge(
        converted_before_df, after_df, on="ID", how="inner"
    )  # 事前アンケートと事後アンケートを横方向に連結
    qa_adult_results_df.to_csv("../data/qa/qa_adult_results.csv", index=False)

    # 児童思春期のアンケートの処理
    igaku_child_before_df = pd.read_csv(igaku_child_before_file_path)
    igaku_parent_before_df = pd.read_csv(
        igaku_parent_before_file_path, skiprows=1
    )  # 1行目には各項目をまとめたヘッダーが書かれているのでスキップする
    igaku_parent_before_df.rename(
        columns={"No": "ID"}, inplace=True
    )  # No列をID列に変更する
    igaku_child_parent_before_df = pd.merge(
        igaku_child_before_df, igaku_parent_before_df, on="ID", how="inner"
    )  # ID列をキーにして横方向に結合する
    extracted_igaku_child_parent_before_df = extract_columns(
        igaku_child_parent_before_df, before_child_parent_columns
    )  # 必要な列を抽出
    converted_child_parent_before_df = convert_child(
        extracted_igaku_child_parent_before_df
    )  # 各項目を数値に変換する
    extracted_igaku_child_after_df = extracted_igaku_after_df[
        ~extracted_igaku_after_df["ID"].isin(igaku_before_df["ID"])
    ]
    qa_child_results_df = pd.merge(
        converted_child_parent_before_df,
        extracted_igaku_child_after_df,
        on="ID",
        how="inner",
    )  # 事前アンケートと事後アンケートを横方向に連結
    qa_child_results_df.to_csv("../data/qa/qa_child_results.csv", index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--riko_before_file_path",
        type=str,
        default="../data/qa/riko_before_raw.csv",
        help="理工学部面談の事前アンケートのファイルパス",
    )
    parser.add_argument(
        "--igaku_before_file_path",
        type=str,
        default="../data/qa/igaku_before_raw.csv",
        help="医学部面談（成人）の事前アンケートのファイルパス",
    )
    parser.add_argument(
        "--igaku_child_before_file_path",
        type=str,
        default="../data/qa/igaku_child_before_raw.csv",
        help="医学部面談（児童思春期）の事前アンケートのファイルパス",
    )
    parser.add_argument(
        "--igaku_parent_before_file_path",
        type=str,
        default="../data/qa/igaku_parent_before_raw.csv",
        help="医学部面談（保護者）の事前アンケートのファイルパス",
    )
    parser.add_argument(
        "--riko_after_file_path",
        type=str,
        default="../data/qa/riko_after_raw.csv",
        help="理工学部面談の事後アンケートのファイルパス",
    )
    parser.add_argument(
        "--igaku_after_file_path",
        type=str,
        default="../data/qa/igaku_after_raw.csv",
        help="医学部面談（全員）の事後アンケートのファイルパス",
    )
    args = parser.parse_args()
    main(
        args.riko_before_file_path,
        args.igaku_before_file_path,
        args.igaku_child_before_file_path,
        args.igaku_parent_before_file_path,
        args.riko_after_file_path,
        args.igaku_after_file_path,
    )
