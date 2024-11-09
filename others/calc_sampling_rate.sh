#!/bin/bash

# 対象ディレクトリを指定 (引数としても渡せる)
DIR=${1:-.}  # 引数が指定されていない場合はカレントディレクトリを使用

# sox がインストールされているか確認
if ! command -v sox &> /dev/null; then
    echo "sox がインストールされていません。インストールしてください。"
    exit 1
fi

# ディレクトリ以下のすべてのwavファイルを再帰的に検索してサンプリングレートを表示
find "$DIR" -type f -name "*.wav" | while read -r file; do
    rate=$(sox --i -r "$file")
    echo "File: $file, Sampling Rate: $rate Hz"
done

