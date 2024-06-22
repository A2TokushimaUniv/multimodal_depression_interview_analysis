import pandas as pd
from audio import analyze_audio
from face import analyze_face
from text import analyze_text
import argparse
from logzero import logger


def main(input_file, output_file, no_text, no_face, no_audio):
    qa_result_df = pd.read_csv(input_file)
    if not no_text:
        qa_result_df = analyze_text(qa_result_df)
    if not no_face:
        qa_result_df = analyze_face(qa_result_df)
    if not no_audio:
        qa_result_df = analyze_audio(qa_result_df)
    qa_result_df.to_csv(output_file, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="../data/preprocessed_data/qa/qa_result.csv",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "--output_file",
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
    input_file = args.input_file
    output_file = args.output_file
    no_text = args.no_text
    no_face = args.no_face
    no_audio = args.no_audio
    logger.info(f"Input_file: {input_file}")
    logger.info(f"Output_file: {output_file}")
    main(input_file, output_file, no_text, no_face, no_audio)
