#!/bin/bash

igaku_dir="../data/voice/igaku"
riko_dir="../data/voice/riko"

igaku_m4a_files=$(find "$igaku_dir" -maxdepth 2 -type f -name "*_zoom_音声_被験者.m4a")
riko_m4a_files=$(find "$riko_dir" -maxdepth 2 -type f -name "audioNLP*.m4a")

output_dir="./elan_output"

# ディレクトリが存在するか確認
if [ ! -d "$output_dir" ]; then
  # 存在しない場合はディレクトリを作成
  mkdir -p "$output_dir"
  echo "Directory created: $output_dir"
fi

for file in $igaku_m4a_files; do
  python3 "elan_segment.py" "$file" "$output_dir/elan_$file.csv"
done

for file in $riko_m4a_files; do
  python3 "elan_segment.py" "$file" "$output_dir/elan_$file.csv"
done