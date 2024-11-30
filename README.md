# Autonomous Vehicle Software - Image Processing

## What is the purpose of the project?

This project is designed for autonomous driving software, which has become increasingly common in recent years. Images from the camera in front of the vehicle are processed in sequence, and a confidence value is used to decide what the vehicle should do based on possible situations. The image processing part is based on the YOLO (You Only Look Once) algorithm. The project aims to be further developed through dataset expansion and algorithm development.

[![Video](https://img.youtube.com/vi/hw_Ltjcw-0Y/0.jpg)](https://www.youtube.com/watch?v=hw_Ltjcw-0Y)

## Algorithm and Models

### YOLO (You Only Look Once)

YOLO is an advanced, real-time object detection algorithm that forms the backbone of this project. It is capable of detecting and classifying multiple objects in a single frame with high accuracy and speed. The project employs YOLO to identify various objects on the road such as pedestrians, vehicles, traffic signs, and road boundaries, all in real-time.

In this project, YOLO is used to detect traffic lights and objects in the lane, which helps in decision-making for the autonomous vehicle. The system evaluates whether the detected objects are in a specific lane or if they pose a potential hazard to the vehicle's movement.

### Road Segmentation Model

To identify the road area and distinguish it from non-road areas in the captured images, a custom-trained road segmentation model is utilized. This model processes the image and outputs a mask that 
highlights the road surface. This mask is then used to focus the vehicle’s decision-making on the road boundaries and lanes, which is crucial for path planning and lane-keeping control.

#### Road Dataset

For the road segmentation task, we use the **Road Detection Segmentation Dataset** available on [Roboflow Universe](https://universe.roboflow.com/lesley-natrop-zgywz/road-detection-segmentation/dataset/10). This dataset contains labeled images for road segmentation, and it is utilized to train the model to identify road boundaries accurately.

You can access the dataset by following the link above and download the dataset for use in training and evaluation.

![Road Segmentation Performance](https://github.com/Eminkorkut/autonomousVehicleSoftware-imageProcessing/blob/main/data/road/results.png)



### Traffic Light Detection Model

Another important component of this project is the detection of traffic lights. The traffic light detection model is trained to recognize the state of traffic lights (green, yellow, or red). This information is vital for the vehicle to make decisions, such as slowing down or stopping if the light is red or proceeding if the light is green.

### Decision-Making System

The decision-making system integrates all the information gathered from the YOLO model, the road segmentation model, and the traffic light detection model. Using this data, the system evaluates the safety of the vehicle's surroundings and computes a driving confidence score. This score helps the vehicle make real-time decisions, such as whether to continue driving, slow down, or stop, based on the detected objects, road conditions, and traffic signals.

The confidence value is continuously updated as the vehicle processes each frame, ensuring that it reacts promptly to any changes in the environment. The driving decision is made based on predefined thresholds, ensuring safe and adaptive driving in various traffic situations.
