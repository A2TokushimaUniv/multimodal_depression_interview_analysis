## 特徴量抽出スクリプト
`preprocess/`でアンケートを前処理した結果に、マルチモーダル特徴量を加えた結果を生成します。

### 実行方法

`python main.py --input_file=qa_file_path --output_file=qa_result_file_path`

`--no_text`, `--no_face`, `--no_audio`を付けることで、必要ないモダリティの特徴量抽出をしないようにもできます。

例えば、言語データの特徴量抽出のみを行いたい場合、`python main.py --input_file=qa_file_path --output_file=qa_result_file_path --no_face --no_audio`とすることで、言語特徴量のみを抽出します。

### 実行結果
アンケートの集計結果（`input_file`）にマルチモーダル特徴量に関する列を追加したCSVファイル（`output_file`）が生成されます。

### 分析方法

#### 言語データ
[東北大 日本語評価極性辞書(名詞編・用言編)](https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html)を`sentiment_polarity/`に格納しています。

これをもとにポジティブ・ネガティブな名詞・用言リストを取得し、GiNZAを使って被験者のテキストの中にどれだけポジティブ・ネガティブな名詞・用言が含まれているかを計算しています。

以下の列が被験者ごとにアンケートの集計結果に追加されます。
- `Pos_Noun_Count`, `Neg_Noun_Count`: ポジティブ・ネガティブな名詞の数
- `Pos_VerbAdj_Count`, `Neg_VerbAdj_Count`: ポジティブ・ネガティブな用言の数
- `Pos_Word_Count`,`Neg_Word_Count`: ポジティブ・ネガティブな名詞・用言の総数
- `Per_Pos_Noun`, `Per_Neg_Noun`: 被験者テキスト中の全名詞のなかのポジティブ・ネガティブな名詞の割合
- `Per_Pos_VerbAdj`,`Per_Neg_VerbAdj`: 被験者テキスト中の全用言のなかのポジティブ・ネガティブな用言の割合

#### 音声データ
音声データをOpenSMILEに入力し、eGeMAPSv02特徴量セットを得ています。

それら特徴量のうち以下の特徴量が被験者ごとにアンケートの集計結果に追加されます。


#### 動画データ
