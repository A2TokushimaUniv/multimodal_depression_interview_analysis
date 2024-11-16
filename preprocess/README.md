# 音声・動画データの前処理スクリプト

## ライブラリのインストール

1. `pip install -r requirements.txt`で必要なライブラリをインストールする
2. [クイックスタート - Reazon Human Interaction Lab](https://research.reazon.jp/projects/ReazonSpeech/quickstart.html)に従い、Reazon Speech のモデルをインストールする


## 処理の流れ

1. 被験者のみが話している音声データから被験者の発話区間を特定する
2. 特定した発話区間をもとに、無音期間を除去した音声データと発話ごとの音声データを生成する
3. 上記の音声データをもとに被験者の発話を Speech-to-Text モデルに入力し、テキストデータを生成する
4. 特定した発話区間をもとに、動画データから被験者が話しているフレームを抽出し、それらを結合した10FPSの動画データを生成する

## 実行手順

1. `../data/raw/video/<igaku> or <riko>/{1..}/.mp4`の形式でデータを格納する
2. `../data/raw/voice/<igaku> or <riko>/{1..}/.m4a`の形式でデータを格納する
3. `python main.py`を実行する

## 実行結果

- 被験者の発話区間のみを抜き出した音声ファイル（全体と発話ごと）及び動画ファイル、被験者の発話テキストが生成される