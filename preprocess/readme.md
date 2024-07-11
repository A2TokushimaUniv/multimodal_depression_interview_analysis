## 前処理スクリプト

### 動画・音声・テキスト

- 1. `data/raw_data/video/<igaku> or <riko>/{1..}/.mp4`の形式でデータを格納する
- 2. `data/raw_data/voice/<igaku> or <riko>/{1..}/.m4a`の形式でデータを格納する
- 3. `preprocess.sh`を実行する

### アンケート

- 1. `data/raw_data/<igaku> or <riko>/.csv`の形式でデータを格納する
- 2. `clean_riko_before.py`を実行して理工アンケートと整形する
- 3. riko, igaku それぞれに対して`convert_qa.py`を実行して、`qa_result.csv`を生成する
