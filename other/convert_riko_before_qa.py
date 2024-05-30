import pandas as pd

riko_file_path = "../data/qa/riko/riko_before_15_clean.xlsx"
riko_df = pd.read_excel(riko_file_path, engine="openpyxl")

# 分析対象ではない列を削除する

# 文章を数値に変換する

# 全角数字を半角数字に直す
