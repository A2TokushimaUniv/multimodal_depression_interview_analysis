#!/bin/bash

igaku_dir="../data/voice/igaku"
riko_dir="../data/voice/riko"

igaku_m4a_files=$(find "$igaku_dir" -maxdepth 2 -type f -name "*_zoom_音声_被験者.m4a")
riko_m4a_files=$(find "$riko_dir" -maxdepth 2 -type f -name "audioNLP*.m4a")

output_dir="./elan_output"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

if [ ! -d "$output_dir/riko" ]; then
  mkdir -p "$output_dir/riko"
fi

if [ ! -d "$output_dir/igaku" ]; then
  mkdir -p "$output_dir/igaku"
fi

for file in $igaku_m4a_files; do
  file_name=$(basename "$file" .m4a)
  output_file_name="elan_$file_name.csv"
  python3 "elan_segment.py" "$file" "$output_dir/igaku/$output_file_name"
done

for file in $riko_m4a_files; do
  file_name=$(basename "$file" .m4a)
  output_file_name="elan_$file_name.csv"
  python3 "elan_segment.py" "$file" "$output_dir/riko/$output_file_name"
done