import pandas as pd

from voice_opensmile import analyze_opensmile_stats, extract_opensmile_lld_features
from voice_wav2vec2 import extract_wav2vec2_features
from voice_vggish import extract_vggish_features
from video_openface import analyze_openface_stats
from video_dlib import extract_dlib_features

from text_ginza import analyze_text
import argparse
from logzero import logger


def main(input_qa_file, input_data_dir, output_qa_file, no_text, no_face, no_audio):
    qa_result_df = pd.read_csv(input_qa_file)
    if not no_text:
        qa_result_df = analyze_text(qa_result_df, input_data_dir)
    if not no_face:
        qa_result_df = analyze_openface_stats(qa_result_df, input_data_dir)
        extract_dlib_features(input_data_dir)
    if not no_audio:
        qa_result_df = analyze_opensmile_stats(qa_result_df, input_data_dir)
        extract_opensmile_lld_features(input_data_dir)
        extract_wav2vec2_features(input_data_dir)
        extract_vggish_features(input_data_dir)
    qa_result_df.to_csv(output_qa_file, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_qa_file",
        default="../data/preprocessed_data/qa/qa_result.csv",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "--input_data_dir",
        default="../data/preprocessed_data",
        type=str,
        help="Path to the input data directory",
    )
    parser.add_argument(
        "--output_qa_file",
        default="../data/preprocessed_data/qa/qa_result_features.csv",
        type=str,
        help="Path to the output file",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        dest="no_text",
        help="Disable text features extraction",
    )
    parser.add_argument(
        "--no_face",
        action="store_true",
        dest="no_face",
        help="Disable face features extraction",
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        dest="no_audio",
        help="Disable audio features extraction",
    )
    args = parser.parse_args()
    input_qa_file = args.input_qa_file
    input_data_dir = args.input_data_dir
    output_qa_file = args.output_qa_file
    no_text = args.no_text
    no_face = args.no_face
    no_audio = args.no_audio
    logger.info(f"Input_file: {input_qa_file}")
    logger.info(f"Output_file: {output_qa_file}")
    main(input_qa_file, input_data_dir, output_qa_file, no_text, no_face, no_audio)
