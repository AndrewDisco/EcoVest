import cv2
import numpy as np
import concurrent.futures
from roboflow import Roboflow
from tqdm import tqdm
from dotenv import load_dotenv
import os
import argparse

# Use Docker to run the inference server locally
# gpu
# docker run -it --rm -p 9001:9001 --gpus all roboflow/roboflow-inference-server-trt
# cpu (arm)
# docker run -it --rm -p 9001:9001  roboflow/roboflow-inference-server-arm-cpu:0.4.4

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("everestv2")
local_inference_server_address = "http://localhost:9001/"
version_number = 1

model = project.version(
    version_number=1,
    local=local_inference_server_address
).model

# Store the last stable detection
last_stable_detection = None

# Function to draw lines from center to nearest prediction and remember the last stable detection


def draw_lines_and_boxes(frame, predictions, image_width, image_height):
    global last_stable_detection

    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    nearest_prediction = None
    nearest_distance = float('inf')

    for prediction in predictions["predictions"]:
        x = int(prediction["x"] * frame.shape[1] / image_width)
        y = int(prediction["y"] * frame.shape[0] / image_height)
        width = int(prediction["width"] * frame.shape[1] / image_width)
        height = int(prediction["height"] * frame.shape[0] / image_height)
        class_name = prediction["class"]
        confidence = prediction["confidence"]

        # Calculate the center of the bounding box
        box_center_x = x
        box_center_y = y

        # Calculate the distance from the frame center to the center of the bounding box
        distance = np.sqrt((center_x - box_center_x)**2 +
                           (center_y - box_center_y)**2)

        # Update nearest prediction if the current one is closer
        if distance < nearest_distance:
            nearest_prediction = prediction
            nearest_distance = distance

        # Draw lines from the center to the center of each bounding box
        if prediction == nearest_prediction:
            color = (0, 0, 255)  # Red color for the nearest box
        else:
            color = (255, 0, 0)  # Blue color for other boxes

        thickness = 2
        cv2.line(frame, (center_x, center_y),
                 (box_center_x, box_center_y), color, thickness)

        # Draw bounding boxes on the frame
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(frame, (x - width // 2, y - height // 2),
                      (x + width // 2, y + height // 2), color, thickness)

        # Add text label with class name and confidence
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x - width // 2, y - height // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Update last_stable_detection if a detection lasted for a bit
    if nearest_prediction is not None:
        last_stable_detection = nearest_prediction

    # Draw the last stable detection with a different color
    if last_stable_detection is not None:
        x = int(last_stable_detection["x"] * frame.shape[1] / image_width)
        y = int(last_stable_detection["y"] * frame.shape[0] / image_height)
        width = int(last_stable_detection["width"]
                    * frame.shape[1] / image_width)
        height = int(last_stable_detection["height"]
                     * frame.shape[0] / image_height)
        color = (0, 0, 255)  # Red color for the last stable detection
        thickness = 2
        cv2.rectangle(frame, (x - width // 2, y - height // 2),
                      (x + width // 2, y + height // 2), color, thickness)

# Function to process each frame and draw lines and boxes


def process_frame(frame, image_width, image_height):
    # Use your model to make predictions on the frame
    predictions = model.predict(frame, confidence=25, overlap=30).json()

    # Draw lines from the center to the nearest prediction and remember the last stable detection
    draw_lines_and_boxes(frame, predictions, image_width, image_height)

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video with object detection")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to the input video")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of processing threads")
    args = parser.parse_args()

    # Load the video
    video_path = args.source
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions and create a VideoWriter for the output
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 30, (frame_width, frame_height))

    # Create a thread pool for parallel processing
    num_threads = args.num_threads

    # Use tqdm to create a progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in tqdm(range(frame_count), desc="Processing video frames"):
            ret, frame = cap.read()
            if not ret:
                break  # Break if the video ends

            # Submit each frame for processing
            processed_frame = process_frame(frame, frame_width, frame_height)
            out.write(processed_frame)  # Write the frame to the output video

    # Release the video capture and output video
    cap.release()
    out.release()

# Release the video capture and output video
cap.release()
out.release()
