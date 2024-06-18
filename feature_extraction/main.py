import pandas as pd
from audio import analyze_opensmile
from face import analyze_openface
from text import analyze_ginza


def main():
    before_sum_df = pd.read_csv("../data/preprocessed_data/qa/before_sum.csv")
    before_sum_df = analyze_openface(before_sum_df)
    before_sum_df = analyze_opensmile(before_sum_df)
    before_sum_df = analyze_ginza(before_sum_df)
    before_sum_df.to_csv(
        "../data/preprocessed_data/qa/before_sum_features.csv", index=False
    )
    return


if __name__ == "__main__":
    main()
