#!/bin/bash

OPENFACE_DIR="../data/feature/openface"
RAW_DIR="../data/raw"

# Ensure OpenFace directory exists
if [ ! -d "$OPENFACE_DIR" ]; then
    echo "Error: OpenFace directory does not exist: $OPENFACE_DIR"
    exit 1
fi

# Iterate through all CSV files in the OpenFace directory
for csv_file in "$OPENFACE_DIR"/*.csv; do
    # Check if the file exists (handle the case where no CSV files are present)
    if [ ! -e "$csv_file" ]; then
        echo "No CSV files found in $OPENFACE_DIR"
        exit 1
    fi

    # Extract the base name of the file (e.g., video_name from video_name.csv)
    base_name=$(basename "$csv_file" .csv)

    # Check if a corresponding video exists in the RAW_DIR
    video_path=$(find "$RAW_DIR" -type f -name "$base_name.mp4" -print -quit)

    if [ -n "$video_path" ]; then
        # Extract the video ID (the directory name containing the video)
        video_id=$(basename $(dirname "$video_path"))

        # Construct the new CSV file name
        new_csv_file="$OPENFACE_DIR/${video_id}.csv"

        # Rename the CSV file
        mv "$csv_file" "$new_csv_file"
        echo "Renamed: $csv_file -> $new_csv_file"
    else
        echo "No matching video found for $csv_file"
    fi
done
