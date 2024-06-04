import pandas as pd


def main():
    igaku_before_sum_file = "../../data/raw_data/qa/igaku/igaku_before_sum.xlsx"
    riko_before_sume_file = "../../data/raw_data/qa/riko/riko_before_sum.xlsx"
    output_file = "../../data/raw_data/qa/before_sum.xlsx"

    igaku_df = pd.read_excel(igaku_before_sum_file, engine="openpyxl")
    riko_df = pd.read_excel(riko_before_sume_file, engine="openpyxl")
    concat_df = pd.concat([igaku_df, riko_df], axis=0)
    concat_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    main()
