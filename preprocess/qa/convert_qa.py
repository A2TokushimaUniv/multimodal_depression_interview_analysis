"""
# 事前アンケートの被験者ごとに各セクションの点数を合計し、
# そのエクセルファイルを生成するスクリプト
"""

import os
import pandas as pd
from logzero import logger


def df_mapping_sum(df, mapping, column_name):
    if mapping:
        for column in df.columns:
            df.loc[:, column] = df[column].map(mapping)
    converted_df = pd.DataFrame(df.sum(axis=1), columns=[column_name])
    return converted_df


def big5_mapping_sum(big5_standard_df, big5_inverse_df, column_name):
    standard_big5_mapping = {
        "全く違うと思う": 1,
        "おおよそ違うと思う": 2,
        "少し違うと思う": 3,
        "どちらでもない": 4,
        "少しそう思う": 5,
        "まあまあそう思う": 6,
        "強くそう思う": 7,
    }
    inverse_big5_mapping = {
        key: 8 - value for key, value in standard_big5_mapping.items()
    }

    big5_standard_df = big5_standard_df.map(standard_big5_mapping)
    big5_inverse_df = big5_inverse_df.map(inverse_big5_mapping)
    big5_df = pd.concat([big5_standard_df, big5_inverse_df], axis=1)
    converted_df = pd.DataFrame(big5_df.sum(axis=1), columns=[column_name])
    return converted_df


def convert_big5(big5_df):
    big5_extrovert_df = big5_df.iloc[:, [0, 5]]
    big5_extrovert_sum_df = big5_mapping_sum(
        big5_extrovert_df.iloc[:, 0], big5_extrovert_df.iloc[:, 1], "BIG5_Extrovert"
    )

    big5_agreeableness_df = big5_df.iloc[:, [1, 6]]
    big5_agreeableness_sum_df = big5_mapping_sum(
        big5_agreeableness_df.iloc[:, 1],
        big5_agreeableness_df.iloc[:, 0],
        "BIG5_Agreeableness",
    )

    big5_conscientiousness_df = big5_df.iloc[:, [2, 7]]
    big5_conscientiousness_sum_df = big5_mapping_sum(
        big5_conscientiousness_df.iloc[:, 0],
        big5_conscientiousness_df.iloc[:, 1],
        "BIG5_Conscientiousness",
    )

    big5_neuroticism_df = big5_df.iloc[:, [3, 8]]
    big5_neuroticism_sum_df = big5_mapping_sum(
        big5_neuroticism_df.iloc[:, 0],
        big5_neuroticism_df.iloc[:, 1],
        "BIG5_Neuroticism",
    )

    big5_openness_df = big5_df.iloc[:, [4, 9]]
    big5_openness_sum_df = big5_mapping_sum(
        big5_openness_df.iloc[:, 0], big5_openness_df.iloc[:, 1], "BIG5_Openness"
    )

    converted_big5_df = pd.concat(
        [
            big5_extrovert_sum_df,
            big5_openness_sum_df,
            big5_neuroticism_sum_df,
            big5_conscientiousness_sum_df,
            big5_agreeableness_sum_df,
        ],
        axis=1,
    )
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
        lambda x: 1 if x >= aq_cutoff_point else 0
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
            return 1
        elif 10 <= score <= 14:
            return 2
        elif 15 <= score <= 21:
            return 3
        else:
            return 0

    converted_gad7_df["GAD7_Level"] = converted_gad7_df["GAD7"].apply(
        categorize_gad7_score
    )
    return converted_gad7_df


def convert_lsas(lsas_df):
    lsas_mapping = {"０": 0, "１": 1, "２": 2, "３": 3}
    converted_lsas_df = df_mapping_sum(lsas_df, lsas_mapping, "LSAS")

    def categorize_lsas_score(score):
        if 30 <= score <= 40:
            return 1
        elif 50 <= score <= 70:
            return 2
        elif score >= 71:
            return 3
        else:
            return 0

    converted_lsas_df["LSAS_Level"] = converted_lsas_df["LSAS"].apply(
        categorize_lsas_score
    )
    return converted_lsas_df


def convert_phq9(phq9_df):
    phq9_cutoff_point = 10
    phq_mapping = {"全くない": 0, "数日": 1, "半分以上": 2, "ほとんど毎日": 3}
    converted_phq9_df = df_mapping_sum(phq9_df, phq_mapping, "PHQ9")
    converted_phq9_df["PHQ9_Flag"] = converted_phq9_df["PHQ9"].apply(
        lambda x: 1 if x >= phq9_cutoff_point else 0
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


def convert_section(big5_df, aq_df, perci_df, gad7_df, lsas_df, phq9_df, sis_df):
    converted_big5_df = convert_big5(big5_df)
    converted_aq_df = convert_aq(aq_df)
    converted_perci_df = convert_perci(perci_df)
    converted_gad7_df = convert_gad7(gad7_df)
    converted_lsas_df = convert_lsas(lsas_df)
    converted_phq9_df = convert_phq9(phq9_df)
    converted_sis_df = convert_sis(sis_df)
    converted_df = pd.concat(
        [
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


def main(before_qa_file, after_qa_file, output_file):
    before_qa_df = pd.read_csv(before_qa_file)
    after_qa_df = pd.read_csv(after_qa_file)
    interview_df = after_qa_df.iloc[:, 1:5]
    interview_df.columns = [
        "Interview_Happy",
        "Interview_Anxiety",
        "Interview_Disgust",
        "Interview_Sad",
    ]

    subject_id_df = before_qa_df.iloc[:, 0]
    subject_id_df = subject_id_df.rename("Subject_ID")
    # 分析対象の列だけを抜き出す
    before_qa_df = before_qa_df.iloc[:, 7:-3]
    # BIG5は10問
    big5_df = before_qa_df.iloc[:, 0:10]
    # AQは50問
    aq_df = before_qa_df.iloc[:, 10:60]
    # PERCIは32問
    perci_df = before_qa_df.iloc[:, 60:92]
    # GAD7は8問
    # GAD7の最後の一問は集計に使わない
    gad7_df = before_qa_df.iloc[:, 92:99]
    # LSASは24×2問
    lsas_df = before_qa_df.iloc[:, 100:148]
    # PHQ9は10問
    # PHQ9の最後の一問は集計に使わない
    phq9_df = before_qa_df.iloc[:, 148:157]
    # SISは6問
    sis_df = before_qa_df.iloc[:, 158:164]
    converted_df = convert_section(
        big5_df, aq_df, perci_df, gad7_df, lsas_df, phq9_df, sis_df
    )
    result_df = pd.concat([subject_id_df, interview_df, converted_df], axis=1)
    result_df.to_csv(output_file, index=False)
    return


if __name__ == "__main__":
    os.makedirs("../../data/preprocessed_data/qa", exist_ok=True)
    riko_before_qa_file = "../../data/raw_data/qa/riko/riko_before_clean.csv"
    riko_after_qa_file = "../../data/raw_data/qa/riko/riko_after.csv"
    riko_output_file = "../../data/preprocessed_data/qa/riko_qa_result.csv"
    logger.info("Input Before QA file: {}".format(riko_before_qa_file))
    logger.info("input After QA file: {}".format(riko_after_qa_file))
    logger.info("Output file: {}".format(riko_output_file))
    main(riko_before_qa_file, riko_after_qa_file, riko_output_file)

    igaku_before_qa_file = "../../data/raw_data/qa/igaku/igaku_before.csv"
    igaku_after_qa_file = "../../data/raw_data/qa/igaku/igaku_after.csv"
    igaku_output_file = "../../data/preprocessed_data/qa/igaku_qa_result.csv"
    logger.info("Input Before QA file: {}".format(igaku_before_qa_file))
    logger.info("input After QA file: {}".format(igaku_after_qa_file))
    logger.info("Output file: {}".format(igaku_output_file))
    main(igaku_before_qa_file, igaku_after_qa_file, igaku_output_file)
