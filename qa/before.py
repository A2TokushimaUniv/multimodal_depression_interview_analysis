import pandas as pd


def _df_mapping_sum(df, mapping, column_name):
    if mapping:
        for column in df.columns:
            df.loc[:, column] = df[column].map(mapping)
    converted_df = pd.DataFrame(df.sum(axis=1), columns=[column_name])
    return converted_df


def _big5_mapping_sum(big5_standard_df, big5_inverse_df, column_name):
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


def _convert_big5(big5_df):
    """
    BIG5を数値に変換する
    """
    big5_extrovert_df = big5_df.iloc[:, [0, 5]]
    big5_extrovert_sum_df = _big5_mapping_sum(
        big5_extrovert_df.iloc[:, 0], big5_extrovert_df.iloc[:, 1], "BIG5_Extrovert"
    )

    big5_agreeableness_df = big5_df.iloc[:, [1, 6]]
    big5_agreeableness_sum_df = _big5_mapping_sum(
        big5_agreeableness_df.iloc[:, 1],
        big5_agreeableness_df.iloc[:, 0],
        "BIG5_Agreeableness",
    )

    big5_conscientiousness_df = big5_df.iloc[:, [2, 7]]
    big5_conscientiousness_sum_df = _big5_mapping_sum(
        big5_conscientiousness_df.iloc[:, 0],
        big5_conscientiousness_df.iloc[:, 1],
        "BIG5_Conscientiousness",
    )

    big5_neuroticism_df = big5_df.iloc[:, [3, 8]]
    big5_neuroticism_sum_df = _big5_mapping_sum(
        big5_neuroticism_df.iloc[:, 0],
        big5_neuroticism_df.iloc[:, 1],
        "BIG5_Neuroticism",
    )

    big5_openness_df = big5_df.iloc[:, [4, 9]]
    big5_openness_sum_df = _big5_mapping_sum(
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


def _convert_aq(aq_df):
    """
    AQ（自閉症傾向）を数値とフラグ（自閉症の有無）に変換する
    """
    aq_cutoff_point = 33
    aq_mapping = {
        "そうである": 1,
        "どちらかといえばそうである": 1,
        "どちらかといえばそうではない": 0,
        "そうではない": 0,
    }
    converted_aq_df = _df_mapping_sum(aq_df, aq_mapping, "AQ")
    # AQ スコアが cutoff_point 以上であれば 1、そうでなければ 0 の列を追加
    converted_aq_df["AQ_Binary"] = converted_aq_df["AQ"].apply(
        lambda x: 1 if x >= aq_cutoff_point else 0
    )
    return converted_aq_df


def _convert_perci(perci_df):
    # PERCIはアンケート上ではもともと数値データなので変換せず合計するだけ
    """
    PERCI（感情制御尺度）を数値に変換する
    """
    converted_perci_df = _df_mapping_sum(perci_df, None, "PERCI")
    return converted_perci_df


def _convert_gad7(gad7_df):
    """
    GAD7（全般不安）を数値と不安のレベルに変換する
    """
    gad7_mapping = {"全くない": 0, "数日": 1, "半分以上": 2, "ほとんど毎日": 3}
    converted_gad7_df = _df_mapping_sum(gad7_df, gad7_mapping, "GAD7")

    def _categorize_gad7_score(score):
        if 5 <= score <= 9:
            return 1
        elif 10 <= score <= 14:
            return 2
        elif 15 <= score <= 21:
            return 3
        else:
            return 0

    converted_gad7_df["GAD7_Level"] = converted_gad7_df["GAD7"].apply(
        _categorize_gad7_score
    )
    return converted_gad7_df


def _convert_lsas(lsas_df):
    """
    LSAS（社交不安）を数値と社交不安のレベルに変換する
    """
    lsas_mapping = {"０": 0, "１": 1, "２": 2, "３": 3}
    converted_lsas_df = _df_mapping_sum(lsas_df, lsas_mapping, "LSAS")

    def _categorize_lsas_score(score):
        if 30 <= score <= 40:
            return 1
        elif 50 <= score <= 70:
            return 2
        elif score >= 71:
            return 3
        else:
            return 0

    converted_lsas_df["LSAS_Level"] = converted_lsas_df["LSAS"].apply(
        _categorize_lsas_score
    )
    return converted_lsas_df


def _convert_phq9(phq9_df):
    """
    PHQ-9を数値とフラグ（鬱状態の有無）に変換する
    """
    phq9_cutoff_point = 10
    phq_mapping = {"全くない": 0, "数日": 1, "半分以上": 2, "ほとんど毎日": 3}
    converted_phq9_df = _df_mapping_sum(phq9_df, phq_mapping, "PHQ9")
    converted_phq9_df["PHQ9_Binary"] = converted_phq9_df["PHQ9"].apply(
        lambda x: 1 if x >= phq9_cutoff_point else 0
    )
    return converted_phq9_df


def _convert_sis(sis_df):
    """
    SIS（自殺念慮）を数値に変換する
    """
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
    converted_sis_df = _df_mapping_sum(sis_df, sui_mapping, "SIS")
    return converted_sis_df


def _convert_scas(scas_df):
    scas_mapping = {
        "いつもそうだ": 3,
        "ときどきそうだ": 2,
        "たまにそうだ": 1,
        "ぜんぜんない": 0,
    }
    converted_scas_df = _df_mapping_sum(scas_df, scas_mapping, "SCAS")
    return converted_scas_df


def _adult_section(big5_df, aq_df, perci_df, gad7_df, lsas_df, phq9_df, sis_df):
    """
    各質問項目を数値に変換する
    """
    converted_big5_df = _convert_big5(big5_df)
    converted_aq_df = _convert_aq(aq_df)
    converted_perci_df = _convert_perci(perci_df)
    converted_gad7_df = _convert_gad7(gad7_df)
    converted_lsas_df = _convert_lsas(lsas_df)
    converted_phq9_df = _convert_phq9(phq9_df)
    converted_sis_df = _convert_sis(sis_df)
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


def _child_section(phq9_df, scas_df, big5_df):
    converted_phq9_df = _convert_phq9(phq9_df)
    converted_scas_df = _convert_scas(scas_df)
    converted_big5_df = _convert_big5(big5_df)
    converted_df = pd.concat(
        [
            converted_phq9_df,
            converted_scas_df,
            converted_big5_df,
        ],
        axis=1,
    )
    return converted_df


def convert_adult(before_qa_df):
    """
    事前アンケート（成人）の各質問項目を数値に変換する
    """
    id_df = before_qa_df.iloc[:, 0]
    before_qa_df = before_qa_df.iloc[:, 1:]
    # BIG5(TIPI-J)は10問
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
    # PHQ9は9問
    phq9_df = before_qa_df.iloc[:, 148:157]
    # SISは6問
    sis_df = before_qa_df.iloc[:, 157:163]
    converted_df = _adult_section(
        big5_df, aq_df, perci_df, gad7_df, lsas_df, phq9_df, sis_df
    )
    result_df = pd.concat([id_df, converted_df], axis=1)
    return result_df


def convert_child(before_qa_df):
    id_df = before_qa_df.iloc[:, 0]
    # 本人回答
    # PHQ9は9問
    phq9_df = before_qa_df.iloc[:, 1:10]
    # SCASは39問
    scas_df = before_qa_df.iloc[:, 10:49]
    # BIG5(TIPI-J)は10問
    big5_df = before_qa_df.iloc[:, 49:59]

    # TODO: とりあえず保護者回答の分析は除く
    # # 保護者回答
    # # AQは50問
    # # TODO: 数値計算に工夫が必要
    # aq_df = before_qa_df.iloc[:, 59:109]
    # aq_df.to_csv("aq_df.csv", index=False)
    # # PERCIは32問
    # # TODO: PERCIの保護者回答は欠損値がある
    # perci_df = before_qa_df.iloc[:, 109:]

    converted_df = _child_section(phq9_df, scas_df, big5_df)
    result_df = pd.concat([id_df, converted_df], axis=1)
    return result_df
