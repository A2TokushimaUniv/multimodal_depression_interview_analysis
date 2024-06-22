#!/bin/bash

# 指定されたディレクトリ内のすべての MP4ファイルの時間の長さを表示するスクリプト

# ディレクトリを引数から取得し、指定がなければ現在のディレクトリを使用
directory=${1:-.}

# 指定されたディレクトリ内のすべての MP4 ファイルを検索
mp4_files=$(find "$directory" -maxdepth 1 -type f -name "*.mp4")

# MP4 ファイルが存在しない場合の処理
if [ -z "$mp4_files" ]; then
  echo "No MP4 files found in the directory: $directory"
  exit 0
fi

# 合計時間を格納する変数とファイル数をカウントする変数
total_seconds=0
file_count=0

# 各 MP4 ファイルの長さを取得して表示
for file in $mp4_files; do
  duration=$(mediainfo --Inform="General;%Duration%" "$file")

  # ミリ秒を秒に変換
  duration_seconds=$((duration / 1000))

  # 秒を時間、分、秒に変換
  hours=$((duration_seconds / 3600))
  minutes=$(((duration_seconds % 3600) / 60))
  seconds=$((duration_seconds % 60))

  echo "File: $(basename "$file"), Duration: ${hours}h ${minutes}mn ${seconds}s"

  # 合計時間に加算
  total_seconds=$((total_seconds + duration_seconds))
  file_count=$((file_count + 1))
done

# 平均時間を計算
if [ $file_count -gt 0 ]; then
  average_seconds=$((total_seconds / file_count))
  average_hours=$((average_seconds / 3600))
  average_minutes=$(((average_seconds % 3600) / 60))
  average_seconds=$((average_seconds % 60))

  # 平均時間を表示
  echo "Average Duration: ${average_hours}h ${average_minutes}mn ${average_seconds}s"
fi

