#!/bin/bash

igaku_voice_dir="../data/raw_data/voice/igaku"
igaku_video_dir="../data/raw_data/video/igaku"
riko_voice_dir="../data/raw_data/voice/riko"
riko_viddeo_dir="../data/raw_data/video/riko"

igaku_m4a_files=$(find "$igaku_voice_dir" -maxdepth 3 -type f -name "*_zoom_音声_被験者.m4a" | sort)
igaku_mp4_files=$(find "$igaku_video_dir" -maxdepth 3 -type f -name "*_zoom_映像・音声*.mp4" | sort)
igaku_m4a_file_array=($igaku_m4a_files)
igaku_mp4_file_array=($igaku_mp4_files)
igaku_file_count=${#igaku_m4a_file_array[@]}

riko_m4a_files=$(find "$riko_voice_dir" -maxdepth 3 -type f -name "audioNLP*.m4a" | sort)
riko_mp4_files=$(find "$riko_viddeo_dir" -maxdepth 3 -type f -name "*-video*.mp4" | sort)
riko_m4a_file_array=($riko_m4a_files)
riko_mp4_file_array=($riko_mp4_files)
riko_file_count=${#riko_m4a_file_array[@]}

output_dir="../data/test_data"


for (( i=0; i<$igaku_file_count; i++ )); do
  # 最初にディレクトリ部分を取得
  dir_path=$(dirname "${igaku_mp4_file_array[$i]}")
  # ディレクトリパスをスラッシュで分割して配列に格納
  IFS='/' read -r -a path_parts <<< "$dir_path"
  # 配列の長さを取得
  path_length=${#path_parts[@]}
  # 後ろから2番目のディレクトリを取得
  dir_num=${path_parts[$((path_length-1))]}
  python3 preprocess.py  --input_video="${igaku_mp4_file_array[$i]}" --input_audio="${igaku_m4a_file_array[$i]}" --output_dir=$output_dir --faculty=igaku --dir_num=$dir_num
done

for (( i=0; i<$riko_file_count; i++ )); do
  # 最初にディレクトリ部分を取得
  dir_path=$(dirname "${riko_mp4_file_array[$i]}")
  # ディレクトリパスをスラッシュで分割して配列に格納
  IFS='/' read -r -a path_parts <<< "$dir_path"
  # 配列の長さを取得
  path_length=${#path_parts[@]}
  # 後ろから2番目のディレクトリを取得
  dir_num=${path_parts[$((path_length-1))]}
  python3 preprocess.py --input_video="${riko_mp4_file_array[$i]}" --input_audio="${riko_m4a_file_array[$i]}" --output_dir=$output_dir --faculty=riko --dir_num=$dir_num
done