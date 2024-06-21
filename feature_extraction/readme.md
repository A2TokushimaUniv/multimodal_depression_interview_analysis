## 特徴量抽出スクリプト
`preprocess/`でアンケートを前処理した結果に、マルチモーダル特徴量を加えた結果を生成します。

### 実行方法

`python main.py --input_file=qa_file_path --output_file=qa_result_file_path`

### 実行結果
アンケートの集計結果（`input_file`）にマルチモーダル特徴量に関する列を追加したCSVファイル（`output_file`）が生成されます。

### 分析方法

#### テキスト
[東北大 日本語評価極性辞書(名詞編・用言編)](https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html)を`sentiment_polarity/`に格納しています。

これをもとにポジティブ・ネガティブな名詞・用言リストを取得し、GiNZAを使って被験者のテキストの中にどれだけポジティブ・ネガティブな名詞・用言が含まれているかを計算しています。

#### 音声


#### 動画