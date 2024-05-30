import pandas as pd

igaku_file_path = "../data/qa/igaku/igaku_before_10.xlsx"  # 医学の事前アンケート
riko_file_path = "../data/qa/riko/riko_before_15.xlsx"  # 理工の事前アンケート
output_file_path = "../data/qa//riko/riko_before_15_clean.xlsx"

igaku_df = pd.read_excel(igaku_file_path, engine="openpyxl")
riko_df = pd.read_excel(riko_file_path, engine="openpyxl")
# 理工学の事前アンケートから下から15行（header含まない）を切り出す
riko_df = riko_df.tail(15)

igaku_headers = igaku_df.columns.tolist()
riko_headers = riko_df.columns.tolist()

# 理工学のアンケートにしかない列を削除
only_riko_headers = set(riko_headers) - set(igaku_headers)
riko_df = riko_df.drop(columns=only_riko_headers)
# 医学アンケートの列の並び順に理工アンケートの列の並びを変更
riko_df = riko_df.reindex(columns=igaku_headers)
cleaned_riko_headers = riko_df.columns.tolist()
assert len(igaku_headers) == len(cleaned_riko_headers)

# 出力
riko_df.to_excel(output_file_path, index=False)
