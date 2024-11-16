#!/bin/bash

# ディレクトリのパス
dir="../data/raw"

# 各被験者のディレクトリをループ
for subject_dir in "$dir"/*/; do
  # mp4ファイルとwavファイルを取得
  mp4_file=$(find "$subject_dir" -type f -name "*.mp4")
  m4a_file=$(find "$subject_dir" -type f -name "*.m4a")

  # ファイルが存在するか確認
  if [[ -f "$mp4_file" && -f "$m4a_file" ]]; then
    # ffprobeを使って各ファイルの長さを秒単位で取得
    mp4_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$mp4_file")
    m4a_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$m4a_file")

    # 小数点以下を丸める
    mp4_duration_rounded=$(printf "%.1f" "$mp4_duration")
    m4a_duration_rounded=$(printf "%.1f" "$m4a_duration")

    # 長さの差を計算
    duration_diff=$(echo "$mp4_duration_rounded - $m4a_duration_rounded" | bc | awk '{print ($1 < 0 ? -$1 : $1)}')

    # 差が1秒以上あるかを確認
    if (( $(echo "$duration_diff >= 1.0" | bc -l) )); then
      echo "不一致: $subject_dir の mp4 と m4a の長さの差が 1 秒以上です (差: $duration_diff 秒)。"
    else
      echo "一致: $subject_dir の mp4 と m4a の長さは 1 秒未満の差です。"
    fi
  else
    echo "警告: $subject_dir に mp4 または m4a ファイルがありません。"
  fi
done

