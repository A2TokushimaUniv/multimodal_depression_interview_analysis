import pandas as pd
from audio import analyze_audio
from face import analyze_face
from text import analyze_text


def main():
    qa_result_df = pd.read_csv("../data/preprocessed_data/qa/qa_result.csv")
    qa_result_df = analyze_face(qa_result_df)
    qa_result_df = analyze_audio(qa_result_df)
    qa_result_df = analyze_text(qa_result_df)
    qa_result_df.to_csv(
        "../data/preprocessed_data/qa/qa_result_features.csv", index=False
    )
    return


if __name__ == "__main__":
    main()
