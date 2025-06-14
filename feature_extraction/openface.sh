#!/bin/bash

openface_feature_extraction_path=$1
video_dir="../data/preprocessed/"

movie_files=$(find "$video_dir" -maxdepth 3 -type f -name "*.mp4" | sort)
movie_files_array=($movie_files)
file_count=${#movie_files_array[@]}

for (( i=0; i<$file_count; i++ )); do
  echo "処理中のファイル: ${movie_files_array[$i]}"
  # Facial Landmark, Action Unit, Gaze, Poseのみを抽出する
  # See: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments#documentation
  # NOTE: 出力ファイル名は指定できないので、出力後にothers/rename_openface_files.shを実行してファイル名を変更する
  "$openface_feature_extraction_path" -f "${movie_files_array[$i]}" -out_dir ../data/feature/openface -2Dfp -aus -pose -gaze
done

echo "全ての処理が完了しました。"
