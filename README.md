# pyFaceDetect

This project leverages the power of OpenCV and dlib for real-time face detection and analysis. It tracks a face (not multiple) in the video feed from your webcam and provides insights into the face's color, landmarks, and more.

## Features

- **Face Detection & Tracking**: Detects and tracks a face in real-time.
- **Color Analysis**: Identifies the face color from predefined ranges.
- **Facial Landmarks**: Marks 68 key points on the face.
- **Various Visualizations**: Shows the face in grayscale, LAB color space, binary, with Canny edges, and a color histogram.

![image](https://github.com/hdunl/pyFaceDetect/assets/54483523/70fda5ce-8d4a-4e42-a44a-42603a4169a7)



## Prerequisites

- Python 3.x
- OpenCV
- dlib
- numpy
- shape_predictor_68_face_landmarks.dat (https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

## How to use

- Clone
- Run main.py, default camera device is used
