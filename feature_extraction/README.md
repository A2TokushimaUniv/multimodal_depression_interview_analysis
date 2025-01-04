# 特徴量抽出スクリプト

`preprocess/`で前処理したデータから特徴量抽出を行い、`qa/`で変換したアンケートに抽出したマルチモーダル特徴量を加えた結果を生成する

## 実行手順

1. 事前に前処理(`preprocess/`)及びアンケートの変換（`qa/`）を実行しておく
2. `python main.py`を実行する
   - なお、`--no_text`, `--no_video`, `--no_voice`を付けることで、必要ないモダリティの特徴量抽出をしないようにもできる
   - 例えば、言語データの特徴量抽出のみを行いたい場合、`python main.py --no_vide --no_voice`とすることで、言語特徴量のみを抽出することができる

## 実行結果

アンケートの集計結果（`input_qa_file`）にマルチモーダル特徴量に関する列を追加した CSV ファイル（`output_qa_file`）が生成される

## 処理の流れ

### 言語データ

[東北大 日本語評価極性辞書(名詞編・用言編)](https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html)を`sentiment_polarity/`に格納している

これをもとにポジティブ・ネガティブな名詞・用言リストを取得し、GiNZA を使って被験者のテキストの中にどれだけポジティブ・ネガティブな名詞・用言が含まれているかを計算している

以下の列が被験者ごとにアンケートの集計結果に追加される

- `Pos_Noun_Count`, `Neg_Noun_Count`: ポジティブ・ネガティブな名詞の数
- `Pos_VerbAdj_Count`, `Neg_VerbAdj_Count`: ポジティブ・ネガティブな用言の数
- `Pos_Word_Count`,`Neg_Word_Count`: ポジティブ・ネガティブな名詞・用言の総数
- `Per_Pos_Noun`, `Per_Neg_Noun`: 被験者テキスト中の全名詞のなかのポジティブ・ネガティブな名詞の割合
- `Per_Pos_VerbAdj`,`Per_Neg_VerbAdj`: 被験者テキスト中の全用言のなかのポジティブ・ネガティブな用言の割合
- `CharPerMinutes`, `WordPerMinutes`: 被験者が 1分間で発話した文字数・単語数

### 音声データ

OpenSMILEとVGGishの2種類の特徴量が得られる

#### OpenSMILE

音声データをOpenSMILE（特徴量セットは`eGeMAPSv02`）に入力し特徴量を得ている

それら特徴量のうち、声のピッチ・大きさ・揺らぎなど、鬱に関連していると思われる特徴量が被験者ごとにアンケートの集計結果に追加される

#### VGGish

音声データをPyTorch実装のVGGish（[harritaylor/torchvggish](https://github.com/harritaylor/torchvggish)）に入力し特徴量を得ている

### 動画データ

OpenFaceのみの特徴量が得られる

#### OpenFace

OpenFaceを用いて顔表情の特徴量を抽出した後に、それらから各Action Unitの強さの平均値・標準偏差を計算している

`main.py`を実行する前に、まず以下の手順で OpenFace を用いて、顔感情の特徴量を抽出しておく必要がある

なお、OpenFaceはDockerイメージを利用して実行しているため、事前にDockerのインストールが必要である
(参考：[Docker · TadasBaltrusaitis/OpenFace Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker))

1. このリポジトリのルートディレクトリに移動する
2. `docker run -v .:/home/openface-build/interview -it algebr/openface:latest`でこのリポジトリをマウントしたDockerイメージを起動する
3. `cd interview/feature_extraction`でリポジトリの特徴量抽出用のディレクトリに移動する
4. `./openface.sh ../../build/bin/FeatureExtraction`でOpenFaceを実行する
