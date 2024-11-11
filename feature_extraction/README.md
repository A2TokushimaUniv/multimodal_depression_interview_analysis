# 特徴量抽出スクリプト

`preprocess/`で前処理したデータから特徴量抽出を行い、アンケートに抽出したマルチモーダル特徴量を加えた結果を生成します。

## 実行方法

1. `pip install -r requirements.txt`で必要なライブラリをインストールする
2. `python main.py`を実行する
   - `--no_text`, `--no_video`, `--no_voice`を付けることで、必要ないモダリティの特徴量抽出をしないようにもできる
   - 例えば、言語データの特徴量抽出のみを行いたい場合、`python main.py --no_vide --no_voice`とすることで、言語特徴量のみを抽出する

## 実行結果

アンケートの集計結果（`input_qa_file`）にマルチモーダル特徴量に関する列を追加した CSV ファイル（`output_qa_file`）が生成されます。

## 言語データ

[東北大 日本語評価極性辞書(名詞編・用言編)](https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html)を`sentiment_polarity/`に格納しています。

これをもとにポジティブ・ネガティブな名詞・用言リストを取得し、GiNZA を使って被験者のテキストの中にどれだけポジティブ・ネガティブな名詞・用言が含まれているかを計算しています。

以下の列が被験者ごとにアンケートの集計結果に追加されます。

- `Pos_Noun_Count`, `Neg_Noun_Count`: ポジティブ・ネガティブな名詞の数
- `Pos_VerbAdj_Count`, `Neg_VerbAdj_Count`: ポジティブ・ネガティブな用言の数
- `Pos_Word_Count`,`Neg_Word_Count`: ポジティブ・ネガティブな名詞・用言の総数
- `Per_Pos_Noun`, `Per_Neg_Noun`: 被験者テキスト中の全名詞のなかのポジティブ・ネガティブな名詞の割合
- `Per_Pos_VerbAdj`,`Per_Neg_VerbAdj`: 被験者テキスト中の全用言のなかのポジティブ・ネガティブな用言の割合
- `CharPerMinutes`, `WordPerMinutes`: 被験者が 1分間で発話した文字数・単語数

## 音声データ

### OpenSMILE

音声データをOpenSMILEに入力し特徴量を得ています。特徴量セットには`eGeMAPSv02`を利用しています。

それら特徴量のうち、声のピッチ・大きさ・揺らぎなど、鬱に関連していると思われる特徴量が被験者ごとにアンケートの集計結果に追加されます。これらの特徴量は後で追加することも可能です。

### VGGish


## 動画データ

### OpenFace

OpenFaceを用いて顔表情の特徴量を抽出した後に、それらから各Action Unitの強さの平均値・標準偏差を計算しています。

`main.py`を実行する前に、まず以下の手順で OpenFace を用いて、顔感情の特徴量を抽出しておく必要があります。

なお、OpenFaceはDockerイメージを利用して実行しているため、事前にDockerのインストールが必要です。
(参考：[Docker · TadasBaltrusaitis/OpenFace Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker))

1. このリポジトリのルートディレクトリに移動する
2. `docker run -v .:/home/openface-build/counseling -it algebr/openface:latest`でこのリポジトリをマウントしたDockerイメージを起動する
3. `cd counseling/feature_extraction`でリポジトリの特徴量抽出用のディレクトリに移動する
4. `./openface.sh ../../build/bin/FeatureExtraction`でOpenFaceを実行する