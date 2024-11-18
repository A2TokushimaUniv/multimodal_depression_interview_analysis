#!/bin/bash

# 対象ディレクトリ（必要に応じて変更）
DIR="../data/feature/openface"
# ディレクトリ内のCSVファイルをループ
for file in "$DIR"/*.csv; do
    # ファイル名から拡張子を除いた部分を取得
    filename=$(basename "$file" .csv)

    # ハイフンで区切られている場合
    if [[ "$filename" == *-* ]]; then
        number=$(echo "$filename" | cut -d'-' -f1)
        newname=$(printf "riko%03d_openface.csv" "$number")

    # アンダーバーで区切られている場合、最初にPがついている場合はPの部分を抜き出す
    elif [[ "$filename" == *_* && "$filename" == P* ]]; then
        number=$(echo "$filename" | cut -d'_' -f1)
        newname="${number}_openface.csv"

    # アンダーバーで区切られている場合、最初にPがついていない場合はCをつけて変換
    elif [[ "$filename" == *_* ]]; then
        number=$(echo "$filename" | cut -d'_' -f1)
        newname="C${number}_openface.csv"

    # その他の形式は無視
    else
        continue
    fi

    # ファイル名の変更
    mv "$file" "$DIR/$newname"
    echo "Renamed $file to $DIR/$newname"
done