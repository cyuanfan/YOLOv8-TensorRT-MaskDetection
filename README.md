# YOLOv8-TensorRT-MaskDetection
Mask detection using YOLOv8 with TensorRT.

# File guidance
1.yolov8_test.py

Can test on video file, webcam or video stream from url.

mode setting:

(1)

tensorrt = 1, program will use .engine file to detect object.

tensorrt = 0, program will use .pt file to detect object.

(2)

mask_detect_mode = 1, program will use mask detection model to detect person wearing mask or not.

mask_detect_mode = 0, program will use official pretrained model to detect 80 classes object.

(3)

webcam = 1, program will detect object from webcam video.

webcam = 0, program will detect object from video file or video stream from url.

2.Detection_app.py

This program provide UI to show object detection from webcam. UI with start/pause button and checkbox 
can enable mask detection or not.

mode setting:

self.b_TensorRTMode = True, program will use .engine file to detect person wearing mask or not.

self.b_TensorRTMode = False, program will use .pt file to detect person wearing mask or not.

(I will add control item in UI for mode change.)

# Inference source
[Video](https://drive.google.com/drive/folders/16zLaimbdfVHhElf467EXvonjqsaBEBnx?usp=drive_link)

# Model
[Model](https://drive.google.com/drive/folders/1IkrbvLPiS0b8fu-ELpd0ViE5tKB9dmrl?usp=drive_link)

# Computer environment

OS:WIN11 / WSL Ubuntu-20.04

NVIDIA GeForce RTX 3060 Laptop

CUDA11.8 + cudnn8.9.6 + TensorRT 8.6 GA

torch=2.1.2+cu118

# Reference
[YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)



