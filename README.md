# EcoVEST AI Part

In our EcoVEST project, we have placed a strong emphasis on developing an innovative solution rooted in the realm of Artificial Intelligence (AI) to address the persistent and pressing issue of waste management in extremely sensitive mountain environments. A core component of this groundbreaking initiative involved the meticulous implementation and training of a cutting-edge AI model, none other than YOLOv5. This model stands out for its exceptional efficiency and precision in detecting objects in images, qualities that are quintessential to dealing with the intricacies of mountainous terrains.

## Table of Contents

- [YOLOv5 Implementation and Training](#yolov5-implementation-and-training)
- [Performance Evaluation](#performance-evaluation)
- [Training Progress Visualization](#training-progress-visualization)
- [Using the Runner File](#using-the-runner-file)
- [Using the Docker Image](#using-the-docker-image)

## YOLOv5 Implementation and Training

To bring this pivotal element of our project to fruition, we embarked on a journey to carefully adapt and train the YOLOv5 model to meet the specific demands of mountain environments. This intricate process entailed architectural adjustments and the optimization of model parameters to ensure the precise identification of various types of waste and debris. From abandoned equipment to plastic packaging and bottles, our model was meticulously trained to identify and pinpoint these objects with unparalleled accuracy.

The bedrock of our model's performance was a comprehensive database comprising real-world images obtained from genuine mountain expeditions and the surrounding regions of Mount Everest. This extensive database encompassed a wide spectrum of scenarios and situations, faithfully reflecting the exacting and challenging realities of mountainous terrains. Thus, our model was primed to confront a multitude of challenges, such as varying light conditions, ever-changing weather, and the rugged, unforgiving landscape.

## Performance Evaluation

In the course of developing and refining our AI model, we harnessed a diverse set of metrics and graphical representations to meticulously evaluate its performance. These metrics played a pivotal role in comprehending the model's evolution and in ensuring it consistently delivered precise results in the domain of mountain waste detection. Here's a detailed exposition of the critical metrics:

- **Train/box_loss**: This metric serves as a yardstick for the loss associated with correctly localizing objects (waste) within images during the training phase. A decrease in this loss serves as a clear indication of an enhancement in object localization.

- **Train/cls_loss**: The train/cls_loss metric gauges the loss linked to the accurate classification of objects within the training images. A decrease in this loss signifies an improved ability of the model to classify different types of waste.

- **Train/dfl_loss**: Train/dfl_loss quantifies the loss associated with the detection of waste objects at various scales within images. A reduction in this loss suggests an augmented adaptability of the model to objects of varying sizes.

- **Metrics/precision(B)**: Precision is a metric that provides insights into the proportion of correct positive detections relative to the total model detections on the test data (the 'B' value relates to the binary metric). A higher precision indicates that the model is prone to making fewer erroneous identifications.

- **Metrics/recall(B)**: Recall represents the proportion of correct positive detections in relation to all positive objects within the test data. A higher recall value signifies that the model is adept at identifying a greater portion of real objects.

- **Val/box_loss**: Analogous to train/box_loss, Val/box_loss measures the loss associated with the correct localization of objects, but on validation data, helping us assess the model's performance on new and unseen data.

- **Val/cls_loss**: Val/cls_loss quantifies the loss tied to accurate object classification, assessed on validation data. A reduction in this loss is an indicator of the model's enhanced classification prowess when dealing with new data.

- **Val/dfl_loss**: Analogous to train/dfl_loss, Val/dfl_loss quantifies the loss associated with object detection but on validation data.

- **Metrics/mAP50(B)**: Mean Average Precision at 50 Intersection over Union (IoU) points (mAP50) serves as a yardstick for the model's overall performance in waste detection. A higher mAP50 value suggests that the model excels in the accurate identification of objects.

- **Metrics/mAP50-95B**: This extends the mAP50 performance evaluation up to 95 IoU points, providing a more granular perspective on the model's precision in diverse scenarios.

The continuous assessment of the model played a pivotal role in honing its capabilities and ensuring it consistently delivered precise and reliable results in the challenging task of mountain waste detection.

![Detection GIF](https://github.com/AndrewDisco/EcoVest/blob/main/images/everestDetect1.gif?raw=true)

## Training Progress Visualization

We would like to offer a comprehensive view of the training progress through the following graphical representation:

![Training Progress](https://github.com/AndrewDisco/EcoVest/blob/main/images/resultscap.png?raw=true)

This visual insight into our model's evolution during the training process underscores our commitment to transparency and accountability.

## Using the Runner File

### Prerequisites

Before using the runner file for object detection, ensure you have the necessary prerequisites installed:

1. Python and required libraries (e.g., OpenCV, NumPy, concurrent.futures, roboflow, tqdm, dotenv).

### Running Object Detection

To run object detection on a video using the provided runner file, follow these steps:

1. Clone or download the project repository to your local machine.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the following command to perform object detection on a video:

   ```bash
   python runner.py --source path/to/input/video.mp4
   ```

   - Replace `path/to/input/video.mp4` with the path to your input video.

4. The script will process the video and create an output video with detected objects. The resulting video will be saved as `output_video.mp4` in the project directory.

5. You can adjust the number of processing threads using the `--num_threads` option. For example, to use 8 processing threads, you can run:

   ```bash
   python runner.py --source path/to/input/video.mp4 --num_threads 8
   ```

## Using the Docker Image

### Prerequisites

Before using the Docker image for object detection, ensure you have Docker installed on your local machine.

### Running the Inference Server Locally

To run the inference server for object detection locally, you can use Docker with the following commands:

#### GPU Version (with GPU support):

```bash
docker run -it --rm -p 9001:9001 --gpus all roboflow/roboflow-inference-server-trt
```

#### CPU (ARM) Version (for ARM-based systems):

```bash
docker run -it --rm -p 9001:9001 roboflow/roboflow-inference-server-arm-cpu:0.4.4
```

Make sure to choose the appropriate Docker command based on your system's hardware.

### API Key Configuration

The Docker image relies on the Roboflow API for object detection. To configure the API key, follow these steps:

1. Load the API key from your .env file. Ensure that you have your API key stored in the .env file in the project directory.

2. The Docker image will use the API key to access the model for object detection.

With these instructions, users will be able to run object detection on videos using the provided runner file and set up the necessary Docker image for inference server locally.

---

For any inquiries or further information, please do not hesitate to reach out to me. Thank you for your interest in the EcoVEST AI project. Your support is invaluable as we strive to make our mountainous environments cleaner and more sustainable.
