#!/bin/bash

openface_feature_extraction_path=$1
riko_video_dir="../data/preprocessed/video/riko"
igaku_video_dir="../data/preprocessed/video/igaku"

riko_movie_files=$(find "$riko_video_dir" -maxdepth 3 -type f -name "*-video*.mp4" | sort)
riko_movie_files_array=($riko_movie_files)
riko_file_count=${#riko_movie_files_array[@]}
igaku_movie_files=$(find "$igaku_video_dir" -maxdepth 3 -type f -name "*_zoom_映像・音声*.mp4" | sort)
igaku_movie_files_array=($igaku_movie_files)
igaku_file_count=${#igaku_movie_files_array[@]}

for (( i=0; i<$riko_file_count; i++ )); do
  "$openface_feature_extraction_path" -f "${riko_movie_files_array[$i]}" -out_dir ../data/preprocessed/openface
done

for (( i=0; i<$igaku_file_count; i++ )); do
  "$openface_feature_extraction_path" -f "${igaku_movie_files_array[$i]}" -out_dir ../data/preprocessed/openface
done
