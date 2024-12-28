# NOTE: 欠損値が増えたら追記する
_MISSING_IDS = ["C020", "C022", "C042", "P019", "riko32"]


def delete_missing_ids(qa_df):
    """
    欠損値のIDを削除する
    """
    # ID列が含まれている場合、_MISSING_IDSに該当するIDを削除
    if "ID" in qa_df.columns:
        qa_df = qa_df[~qa_df["ID"].isin(_MISSING_IDS)]
    return qa_df


def is_csv(file_paths):
    """
    CSVファイルかどうかを判定する
    """
    for file_path in file_paths:
        if file_path.endswith(".csv"):
            return True
        else:
            return False


def add_riko_id(riko_df):
    """
    理工学部のアンケート結果にはID列がないので追加する
    """
    if "ID" not in riko_df.columns:
        riko_df["ID"] = [f"riko{i:02}" for i in range(1, len(riko_df) + 1)]
    return riko_df


def extract_columns(target_df, columns_df):
    """
    指定した列のみを抽出する
    """
    common_columns = [col for col in columns_df.columns if col in target_df.columns]
    return target_df[common_columns]
