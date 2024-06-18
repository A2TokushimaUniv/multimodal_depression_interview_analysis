"""
# 数値化した事前アンケートの連結
"""

import pandas as pd


def main():
    igaku_qa_result_file = "../../data/raw_data/qa/igaku/igaku_qa_result.csv"
    riko_qa_result_file = "../../data/raw_data/qa/riko/riko_qa_result.csv"
    output_file = "../../data/preprocessed_data/qa/qa_result.csv"

    igaku_df = pd.read_csv(igaku_qa_result_file)
    riko_df = pd.read_csv(riko_qa_result_file)
    concat_df = pd.concat([igaku_df, riko_df], axis=0)
    concat_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
