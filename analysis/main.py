import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from logzero import logger

questionnaire_columns = [
    "Interview_Happy",
    "Interview_Anxiety",
    "Interview_Disgust",
    "Interview_Sad",
    "BIG5_Extrovert",
    "BIG5_Openness",
    "BIG5_Neuroticism",
    "BIG5_Conscientiousness",
    "BIG5_Agreeableness",
    "AQ",
    "PERCI",
    "GAD7",
    "LSAS",
    "PHQ9",
    "SIS",
    "SCAS",
]

au_columns = [
    f"AU{'0' if au < 10 else ''}{au}_r_{stat}".strip()
    for au in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    for stat in ["Mean", "Stddev"]
]
face_feature_columns = au_columns + ["AUall_r_Mean", "AUall_r_Stddev"]

text_feature_columns = [
    "Neg_Noun_Count",
    "Neg_VerbAdj_Count",
    "Neg_Word_Count",
    "Pos_Noun_Count",
    "Pos_VerbAdj_Count",
    "Pos_Word_Count",
    "Per_Pos_Noun",
    "Per_Pos_VerbAdj",
    "Per_Neg_Noun",
    "Per_Neg_VerbAdj",
    "CharPerMinutes",
    "WordPerMinutes",
]

voice_feature_columns = [
    "PitchMean",
    "PitchStddev",
    "LoudnessMean",
    "LoudnessStddev",
    "JitterMean",
    "JitterStddev",
    "ShimmerMean",
    "ShimmerStddev",
    "HNRdBACF",
    "F0semitone",
    "F3frequency",
]


def get_significant_pairs(output_file_path, correlation_matrix, threshold):
    """
    相関係数の絶対値が閾値以上のペアを見つける
    """
    corr_pairs = correlation_matrix.unstack()
    significant_pairs = corr_pairs[abs(corr_pairs) >= threshold]
    significant_pairs = significant_pairs[
        significant_pairs.index.get_level_values(0)
        != significant_pairs.index.get_level_values(1)
    ]
    significant_pairs = significant_pairs.drop_duplicates()

    # p値を計算し、辞書に格納
    p_values = {}
    for pair in significant_pairs.index:
        col1, col2 = pair
        _, p_value = stats.pearsonr(correlation_matrix[col1], correlation_matrix[col2])
        p_values[pair] = p_value

    # 結果をデータフレームに変換
    significant_pairs_df = pd.DataFrame(significant_pairs, columns=["correlation"])
    significant_pairs_df["p_value"] = significant_pairs_df.index.map(p_values)

    # インデックスをカラムに変換
    significant_pairs_df.reset_index(inplace=True)
    significant_pairs_df.columns = ["Feature1", "Feature2", "correlation", "p_value"]

    multimodal_feature_columns = (
        face_feature_columns + text_feature_columns + voice_feature_columns
    )

    # questionnaire_columns同士のペア、multimodal_feature_columns同士のペアを削除
    significant_pairs_df = significant_pairs_df[
        ~(
            (
                significant_pairs_df["Feature1"].isin(questionnaire_columns)
                & significant_pairs_df["Feature2"].isin(questionnaire_columns)
            )
            | (
                significant_pairs_df["Feature1"].isin(multimodal_feature_columns)
                & significant_pairs_df["Feature2"].isin(multimodal_feature_columns)
            )
        )
    ]

    significant_pairs_df.to_csv(output_file_path, index=False)


def get_heatmap(output_file_path, correlation_matrix):
    """
    相関行列のヒートマップをプロットする
    """
    logger.info("ヒートマップを生成しています...")
    plt.figure(figsize=(50, 30))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
    )
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(output_file_path)


def calculate_statistics(output_file_path, df):
    """
    特徴量の平均と標準偏差を計算する
    """
    means = df.mean()
    stds = df.std()

    results = pd.DataFrame({"Mean": means, "Standard Deviation": stds})

    results.to_csv(output_file_path, index=True)


def main(input_file_path, output_dir, threshold):
    qa_df = pd.read_csv(input_file_path)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("相関係数を計算しています...")
    # アンケートデータから相関分析をしない列を削除
    columns_to_exclude = [
        col for col in qa_df.columns if "Level" in col or "Binary" in col or "ID" == col
    ]
    qa_df = qa_df.drop(columns=columns_to_exclude)
    file_suffix = str(threshold).replace(".", "")
    calculate_statistics(
        os.path.join(output_dir, "qa_columns_statistics.csv"),
        qa_df.filter(items=(questionnaire_columns)),
    )

    # テキスト特徴量の相関分析
    os.makedirs(os.path.join(output_dir, "text"), exist_ok=True)
    text_df = qa_df.filter(items=(questionnaire_columns + text_feature_columns))
    calculate_statistics(
        os.path.join(output_dir, "text", "text_columns_statistics.csv"),
        text_df[text_feature_columns],
    )
    text_correlation_matrix = text_df.corr()
    get_heatmap(
        os.path.join(output_dir, "text", "text_correlation_matrix_heatmap.png"),
        text_correlation_matrix,
    )
    get_significant_pairs(
        os.path.join(output_dir, "text", f"text_significant_pairs_{file_suffix}.csv"),
        text_correlation_matrix,
        threshold,
    )

    # 音声特徴量の相関分析
    os.makedirs(os.path.join(output_dir, "voice"), exist_ok=True)
    voice_df = qa_df.filter(items=(questionnaire_columns + voice_feature_columns))
    calculate_statistics(
        os.path.join(output_dir, "voice", "voice_columns_statistics.csv"),
        voice_df[voice_feature_columns],
    )
    voice_correlation_matrix = voice_df.corr()
    get_heatmap(
        os.path.join(output_dir, "voice", "voice_correlation_matrix_heatmap.png"),
        voice_correlation_matrix,
    )
    get_significant_pairs(
        os.path.join(output_dir, "voice", f"voice_significant_pairs_{file_suffix}.csv"),
        voice_correlation_matrix,
        threshold,
    )

    # 顔特徴量の相関分析
    os.makedirs(os.path.join(output_dir, "face"), exist_ok=True)
    face_df = qa_df.filter(items=(questionnaire_columns + face_feature_columns))
    calculate_statistics(
        os.path.join(output_dir, "face", "face_columns_statistics.csv"),
        face_df[face_feature_columns],
    )
    face_correlation_matrix = face_df.corr()
    get_heatmap(
        os.path.join(output_dir, "face", "face_correlation_matrix_heatmap.png"),
        face_correlation_matrix,
    )
    get_significant_pairs(
        os.path.join(output_dir, "face", f"face_significant_pairs_{file_suffix}.csv"),
        face_correlation_matrix,
        threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adult_qa_file_path",
        default="../data/qa/adult_features.csv",
        help="分析対象のアンケート（成人）",
        type=str,
    )
    parser.add_argument(
        "--child_qa_file_path",
        default="../data/qa/child_features.csv",
        help="分析対象のアンケート（児童思春期）",
        type=str,
    )
    parser.add_argument(
        "--output_adult_dir",
        default="./adult_analysis_results",
        help="成人アンケートの分析結果の出力ディレクトリ",
        type=str,
    )
    parser.add_argument(
        "--output_child_dir",
        default="./child_analysis_results",
        help="児童思春期アンケートの分析結果の出力ディレクトリ",
        type=str,
    )
    parser.add_argument(
        "--threshold", default=0.1, help="出力する相関係数の閾値", type=float
    )
    args = parser.parse_args()
    main(args.adult_qa_file_path, args.output_adult_dir, args.threshold)
    main(args.child_qa_file_path, args.output_child_dir, args.threshold)
