from voice_opensmile import extract_opensmile_lld_feature, analyze_opensmile_stats
from video_openface import analyze_openface_stats
from voice_vggish import extract_vggish_feature
from text_ginza import analyze_text
import pandas as pd
import argparse
from logzero import logger


def main(
    input_adult_qa_file,
    input_child_qa_file,
    preprocessed_dir,
    feature_dir,
    output_adult_qa_file,
    output_child_qa_file,
    no_text,
    no_video,
    no_voice,
):
    logger.info("特徴量の抽出を開始します")
    adult_qa_df = pd.read_csv(input_adult_qa_file)
    child_qa_df = pd.read_csv(input_child_qa_file)
    if not no_text:
        logger.info("テキスト特徴量を抽出しています....")
        adult_qa_df, child_qa_df = analyze_text(
            adult_qa_df, child_qa_df, preprocessed_dir, feature_dir
        )
    if not no_video:
        logger.info("動画特徴量を抽出しています....")
        adult_qa_df, child_qa_df = analyze_openface_stats(
            adult_qa_df, child_qa_df, feature_dir
        )
    if not no_voice:
        logger.info("音声特徴量を抽出しています....")
        adult_qa_df, child_qa_df = analyze_opensmile_stats(
            adult_qa_df, child_qa_df, preprocessed_dir
        )
        extract_opensmile_lld_feature(preprocessed_dir, feature_dir)
        extract_vggish_feature(preprocessed_dir, feature_dir)
    adult_qa_df.to_csv(output_adult_qa_file, index=False)
    child_qa_df.to_csv(output_child_qa_file, index=False)
    logger.info("特徴量の抽出が完了しました")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_adult_qa_file",
        default="../data/qa/adult_results.csv",
        type=str,
        help="入力する成人のアンケート結果へのファイルパス",
    )
    parser.add_argument(
        "--input_child_qa_file",
        default="../data/qa/child_results.csv",
        type=str,
        help="入力する児童思春期のアンケート結果へのファイルパス",
    )
    parser.add_argument(
        "--preprocessed_dir",
        default="../data/preprocessed",
        type=str,
        help="前処理したデータを格納しているディレクトリ",
    )
    parser.add_argument(
        "--feature_dir",
        default="../data/feature",
        type=str,
        help="抽出する特徴量を保存するディレクトリ",
    )
    parser.add_argument(
        "--output_adult_feature_file",
        default="../data/qa/adult_features.csv",
        type=str,
        help="出力する成人のアンケート結果へのファイルパス",
    )
    parser.add_argument(
        "--output_child_feature_file",
        default="../data/qa/child_features.csv",
        type=str,
        help="出力する児童思春期のアンケート結果へのファイルパス",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        dest="no_text",
        help="テキスト特徴量を抽出するか否か",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        dest="no_video",
        help="動画特徴量を抽出するか否か",
    )
    parser.add_argument(
        "--no_voice",
        action="store_true",
        dest="no_voice",
        help="音声特徴量を抽出するか否か",
    )

    args = parser.parse_args()
    input_adult_qa_file = args.input_adult_qa_file
    input_child_qa_file = args.input_child_qa_file
    preprocessed_dir = args.preprocessed_dir
    feature_dir = args.feature_dir
    output_adult_feature_file = args.output_adult_feature_file
    output_child_feature_file = args.output_child_feature_file
    no_text = args.no_text
    no_video = args.no_video
    no_voice = args.no_voice

    logger.info(f"入力アンケートデータ（成人）: {input_adult_qa_file}")
    logger.info(f"入力アンケートデータ（児童思春期）: {input_child_qa_file}")
    logger.info(f"前処理済みデータディレクトリ: {preprocessed_dir}")
    logger.info(f"特徴量ディレクトリ: {feature_dir}")
    logger.info(f"出力アンケートデータ（成人）: {output_adult_feature_file}")
    logger.info(f"出力アンケートデータ（児童思春期）: {output_child_feature_file}")
    main(
        input_adult_qa_file,
        input_child_qa_file,
        preprocessed_dir,
        feature_dir,
        output_adult_feature_file,
        output_child_feature_file,
        no_text,
        no_video,
        no_voice,
    )
