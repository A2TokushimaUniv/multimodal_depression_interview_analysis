import cv2
from deepface import DeepFace
import json


def analyze_video(video_path, output_path, frame_interval=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Analyze the frame using DeepFace
            try:
                vector = DeepFace.represent(frame, model_name="OpenFace")
                print(vector[0]["embedding"])
            except Exception as e:
                print(f"Error analyzing frame {frame_count}: {e}")
        frame_count += 1
    cap.release()
    print("Video analysis complete.")
    # Write results to the output file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")


# Example usage
video_path = "./data/preprocessed_data/video/riko/1/utterance_18.mp4"
output_path = "output_analysis.json"
analyze_video(video_path, output_path)
