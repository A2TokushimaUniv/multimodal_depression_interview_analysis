"""
# 理工学の事前アンケートにテスト時のデータが入っていて汚かったため、
# 医学の事前アンケートに合わせて修正するスクリプト
"""

import pandas as pd


def main():
    igaku_file = "../../data/qa/igaku_before.csv"  # 医学の事前アンケート
    riko_file = "../../data/qa/riko_before.csv"  # 理工の事前アンケート
    output_file = "../../data/qa/riko_before_clean.csv"

    igaku_df = pd.read_csv(igaku_file)
    riko_df = pd.read_csv(riko_file)

    # 理工学の事前アンケートから下から15行（header含まない）を切り出す
    riko_df = riko_df.tail(15)

    igaku_headers = igaku_df.columns.tolist()
    riko_headers = riko_df.columns.tolist()

    # 理工学のアンケートにしかない列を削除
    only_riko_headers = set(riko_headers) - set(igaku_headers)
    riko_df = riko_df.drop(columns=only_riko_headers)
    # 医学アンケートの列の並び順に理工アンケートの列の並びを変更
    riko_df = riko_df.reindex(columns=igaku_headers)
    cleaned_riko_headers = riko_df.columns.tolist()
    assert len(igaku_headers) == len(cleaned_riko_headers)

    # 出力
    riko_df.to_csv(output_file, index=False)
    return


if __name__ == "__main__":
    main()
