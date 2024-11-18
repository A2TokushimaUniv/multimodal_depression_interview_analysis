#!/bin/bash

openface_feature_extraction_path=$1
video_dir="../data/preprocessed/"

movie_files=$(find "$video_dir" -maxdepth 3 -type f -name "*.mp4" | sort)
movie_files_array=($movie_files)
file_count=${#movie_files_array[@]}

for (( i=0; i<$file_count; i++ )); do
  echo "処理中のファイル: ${movie_files_array[$i]}"
  # Facial LandmarkとAction Unitのみを抽出する
  # NOTE: 出力ファイル名は指定できないので、出力後に<ID名>.csvとなるように手動で修正する
  "$openface_feature_extraction_path" -f "${movie_files_array[$i]}" -out_dir ../data/feature/openface -2Dfp -aus
done

echo "全ての処理が完了しました。"