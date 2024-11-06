# DriverDrowsiness

## Description
A real-time driver drowsiness detection system using computer vision and deep learning. This project was created as the Honours year project in the department of applied mathematics at Stellenbosch university. The project detects the driver features in real-time decides on the drowsiness level of the person with a self-trained convolutional neural network.
The project may be readily installed and many improved upon.

## Table of Contents
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)

## Demo

![Demo of the project](DDD.gif)

## Installation
### Prerequisites
- Python 3.10.x - 3.12.x
- Webcam access
### Installation Steps
1. Clone the repository
```bash
git clone https://github.com/Andreas-Wild/DriverDrowsiness.git
```
2.  Set the src folder as the working directory
```bash
cd DriverDrowsiness/src
```
3. Install the necessary requirements. (Preferably in a virtual environment)
```bash
pip install -r requirements.txt
```
## Usage
Ensure that the src folder is set as the working directory
```bash
  python webcam.py
```
A window will open that diplays the webcam view. A bounding box along with eye positions will be annotated on any detected faces in the frame. To quit the program press the 'q' key.

## Configuration

- The `ALERT_THRESHOLD` variable in the preamble of the `webcam.py` file may be adjusted to set the threshold of the drowsiness counter. This value is measured in seconds.
- The `coin` variable may be adjusted in switch between efficiency and accuracy modes. This variable determines whether both eyes are analysed at each frame position.
- To remove screen annotations the the `verbose` parameter in the `find_eyes` function may be set to `false`.
- The decision CNN may be readily adjusted and is modular. Provided that the input is a 64x64 greyscale image, the output is a float between (0,1).

## Acknowledgements
- The face detection pipeline uses the YuNet model found here. ([YuNet](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx))
- A big thank you to the depratment of applied mathematics at Stellenbosch university!
