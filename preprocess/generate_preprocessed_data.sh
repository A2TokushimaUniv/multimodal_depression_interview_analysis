#!/bin/bash

igaku_voice_dir="../data/raw_data/voice/igaku"
igaku_video_dir="../data/raw_data/video/igaku"
riko_voice_dir="../data/raw_data/voice/riko"
riko_viddeo_dir="../data/raw_data/video/riko"

igaku_m4a_files=$(find "$igaku_voice_dir" -maxdepth 2 -type f -name "*_zoom_音声_被験者.m4a" | sort)
igaku_mp4_files=$(find "$igaku_video_dir" -maxdepth 2 -type f -name "*_zoom_映像・音声*.mp4" | sort)
igaku_m4a_file_array=($igaku_m4a_files)
igaku_mp4_file_array=($igaku_mp4_files)
igaku_file_count=${#igaku_m4a_file_array[@]}

riko_m4a_files=$(find "$riko_voice_dir" -maxdepth 2 -type f -name "audioNLP*.m4a" | sort)
riko_mp4_files=$(find "$riko_viddeo_dir" -maxdepth 2 -type f -name "*-video*.mp4" | sort)
riko_m4a_file_array=($riko_m4a_files)
riko_mp4_file_array=($riko_mp4_files)
riko_file_count=${#riko_m4a_file_array[@]}

output_dir="../data/preprocessed_data"


for (( i=0; i<$igaku_file_count; i++ )); do
  python3 preprocess.py "${igaku_m4a_file_array[$i]}" "${igaku_mp4_file_array[$i]}" --output_dir=$output_dir --faculty=igaku
done


for (( i=0; i<$riko_file_count; i++ )); do
  python3 preprocess.py "${riko_m4a_file_array[$i]}" "${riko_mp4_file_array[$i]}" --output_dir=$output_dir --faculty=riko
done

