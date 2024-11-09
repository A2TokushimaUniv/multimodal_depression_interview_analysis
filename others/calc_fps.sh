#!/bin/bash

# 引数でディレクトリパスを取得
DIR_PATH="$1"

# 引数が空の場合はエラーメッセージを表示して終了
if [ -z "$DIR_PATH" ]; then
  echo "Usage: $0 /path/to/your/directory"
  exit 1
fi

# 合計FPSとファイル数を初期化
total_fps=0
file_count=0

# 再帰的にMP4ファイルを検索してループ処理
find "$DIR_PATH" -type f -name "*.mp4" | while read -r file; do
  # FPSを取得して計算 (r_frame_rate の分数を実数に変換)
  fps=$(ffprobe -v 0 -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$file" | awk -F/ '{if ($2) print $1/$2; else print $1}')
  
  # FPSが取得できた場合、合計に加算しファイル数をインクリメント
  if [ ! -z "$fps" ]; then
    total_fps=$(echo "$total_fps + $fps" | bc)
    file_count=$((file_count + 1))
    echo "File: $file - FPS: $fps"
  fi
done

# 平均を計算
if [ $file_count -gt 0 ]; then
  average_fps=$(echo "scale=2; $total_fps / $file_count" | bc)
  echo "Average FPS: $average_fps"
else
  echo "No MP4 files found or unable to calculate FPS."
fi

