import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSVファイルの読み込み
file_path = "../data/preprocessed_data/qa/before_sum_features.csv"
df = pd.read_csv(file_path)

# タイムスタンプとLevelとFlagに関する列を削除
columns_to_exclude = [
    col
    for col in df.columns
    if "Level" in col or "Flag" in col or "タイムスタンプ" == col
]
df = df.drop(columns=columns_to_exclude)

print(df.head())

# 相関係数を計算
correlation_matrix = df.corr()

# 相関係数行列の表示
print(correlation_matrix)
correlation_matrix.to_csv("correlation_matrix.csv")

# ヒートマップの作成と表示
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Correlation Matrix Heatmap")

# ヒートマップの画像を保存
plt.savefig("correlation_matrix_heatmap.png")

# # ヒートマップの表示（省略可能）
# plt.show()
