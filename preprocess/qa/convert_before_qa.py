"""
# 事前アンケートの被験者ごとに各セクションの点数を合計し、
# そのエクセルファイルを生成するスクリプト
"""

import pandas as pd
import argparse
from logzero import logger


def df_mapping_sum(df, mapping, column_name):
    if mapping:
        for column in df.columns:
            df[column] = df[column].map(mapping)
    converted_df = pd.DataFrame(df.sum(axis=1), columns=[column_name])
    return converted_df


def convert_big5(big5_df):
    big5_mapping = {
        "全く違うと思う": 1,
        "おおよそ違うと思う": 2,
        "少し違うと思う": 3,
        "どちらでもない": 4,
        "少しそう思う": 5,
        "まあまあそう思う": 6,
        "強くそう思う": 7,
    }
    converted_big5_df = df_mapping_sum(big5_df, big5_mapping, "BIG5")
    return converted_big5_df


def convert_aq(aq_df):
    aq_cutoff_point = 33
    aq_mapping = {
        "そうである": 1,
        "どちらかといえばそうである": 1,
        "どちらかといえばそうではない": 0,
        "そうではない": 0,
    }
    converted_aq_df = df_mapping_sum(aq_df, aq_mapping, "AQ")
    # AQ スコアが cutoff_point 以上であれば 1、そうでなければ 0 の列を追加
    converted_aq_df["AQ_Flag"] = converted_aq_df["AQ"].apply(
        lambda x: "あり" if x >= aq_cutoff_point else "なし"
    )
    return converted_aq_df


def convert_perci(perci_df):
    # PERCIはアンケート上ではもともと数値データなので変換せず合計するだけ
    converted_perci_df = df_mapping_sum(perci_df, None, "PERCI")
    return converted_perci_df


def convert_gad7(gad7_df):
    gad7_mapping = {"全くない": 0, "数日": 1, "半分以上": 2, "ほとんど毎日": 3}
    converted_gad7_df = df_mapping_sum(gad7_df, gad7_mapping, "GAD7")

    def categorize_gad7_score(score):
        if 5 <= score <= 9:
            return "軽度"
        elif 10 <= score <= 14:
            return "中等度"
        elif 15 <= score <= 21:
            return "重度"
        else:
            return "なし"

    converted_gad7_df["GAD7_Level"] = converted_gad7_df["GAD7"].apply(
        categorize_gad7_score
    )
    return converted_gad7_df


def convert_lsas(lsas_df):
    lsas_mapping = {"０": 0, "１": 1, "２": 2, "３": 3}
    converted_lsas_df = df_mapping_sum(lsas_df, lsas_mapping, "LSAS")

    def categorize_lsas_score(score):
        if 30 <= score <= 40:
            return "境界域"
        elif 50 <= score <= 70:
            return "中等症"
        else:
            return "なし"

    converted_lsas_df["LSAS_Level"] = converted_lsas_df["LSAS"].apply(
        categorize_lsas_score
    )
    return converted_lsas_df


def convert_phq9(phq9_df):
    phq9_cutoff_point = 10
    phq_mapping = {"全くない": 0, "数日": 1, "半分以上": 2, "ほとんど毎日": 3}
    converted_phq9_df = df_mapping_sum(phq9_df, phq_mapping, "PHQ9")
    converted_phq9_df["PHQ9_Flag"] = converted_phq9_df["PHQ9"].apply(
        lambda x: "あり" if x >= phq9_cutoff_point else "なし"
    )
    return converted_phq9_df


def convert_sis(sis_df):
    sui_mapping = {
        "全く思わない": 0,
        "あまり強く思わない": 1,
        "強く思う": 2,
        "生きたいという気持ちが死にたいという気持ちよりも強い": 0,
        "両者が同じ程度である": 1,
        "死にたいという気持ちが生きたいという気持ちよりも強い": 2,
        "全くない": 0,
        "ややある": 1,
        "中程度以上にある": 2,
        "自殺したいとは思わない、または思っても短く、すぐに過ぎ去る": 0,
        "長く続く": 1,
        "持続的（慢性的）に続く": 2,
        "全く起こらない、または起こったとしても稀にしかない": 0,
        "たまにある": 1,
        "繰り返し生じる、または持続的に起こる": 2,
        "考えていない": 0,
        "考えてみたが、それほど詳しくは考えていない": 1,
        "詳しく考え、十分に練られている": 2,
    }
    converted_sis_df = df_mapping_sum(sis_df, sui_mapping, "SIS")
    return converted_sis_df


def convert_section(
    timestamp_df, big5_df, aq_df, perci_df, gad7_df, lsas_df, phq9_df, sis_df
):
    converted_big5_df = convert_big5(big5_df)
    converted_aq_df = convert_aq(aq_df)
    converted_perci_df = convert_perci(perci_df)
    converted_gad7_df = convert_gad7(gad7_df)
    converted_lsas_df = convert_lsas(lsas_df)
    converted_phq9_df = convert_phq9(phq9_df)
    converted_sis_df = convert_sis(sis_df)
    converted_df = pd.concat(
        [
            timestamp_df,
            converted_big5_df,
            converted_aq_df,
            converted_perci_df,
            converted_gad7_df,
            converted_lsas_df,
            converted_phq9_df,
            converted_sis_df,
        ],
        axis=1,
    )
    return converted_df


def main(qa_file_path, output_file_path):
    qa_df = pd.read_excel(qa_file_path, engine="openpyxl")
    timestamp_df = qa_df.iloc[:, 0]
    # 分析対象の列だけを抜き出す
    qa_df = qa_df.iloc[:, 7:-3]
    # BIG5は10問
    big5_df = qa_df.iloc[:, 0:10]
    # AQは50問
    aq_df = qa_df.iloc[:, 10:60]
    # PERCIは32問
    perci_df = qa_df.iloc[:, 60:92]
    # GAD7は8問
    # GAD7の最後の一問は集計に使わない
    gad7_df = qa_df.iloc[:, 92:99]
    # LSASは24×2問
    lsas_df = qa_df.iloc[:, 100:148]
    # PHQ9は10問
    # PHQ9の最後の一問は集計に使わない
    phq9_df = qa_df.iloc[:, 148:157]
    # SISは6問
    sis_df = qa_df.iloc[:, 158:164]
    converted_df = convert_section(
        timestamp_df, big5_df, aq_df, perci_df, gad7_df, lsas_df, phq9_df, sis_df
    )
    print(converted_df)
    converted_df.to_excel(output_file_path, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_qa_file", help="Path to the QA file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    qa_file_path = args.input_qa_file
    output_file_path = args.output_file
    logger.info("Input QA file: {}".format(qa_file_path))
    logger.info("Output file: {}".format(output_file_path))
    main(qa_file_path, output_file_path)
