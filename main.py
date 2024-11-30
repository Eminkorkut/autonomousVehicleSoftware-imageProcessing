from ultralytics import YOLO
import numpy as np
import argparse
import math
import cv2

from side import detect

def argsRun(
    source='data/videoRoad1.mp4',
    yoloModelSource='data/yolo11s.pt',
    roadModelSource='data/road/best.pt',
    trafficLightModelSource='data/trafficLight.pt',
    outputVideoPath='output_video.mp4'
):
    lineColor = (0, 0, 0)
    classNames = ["green", "red", "yellow"]

    # Load required models
    try:
        roadModel = YOLO(roadModelSource)
        trafficLightModel = YOLO(trafficLightModelSource)
        yoloModel = YOLO(yoloModelSource)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Open the video source
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        print(f"Error opening video file: {source}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(outputVideoPath, fourcc, fps, (frameWidth, frameHeight))

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("The video ended or could not be read.")
            break

        try:
            roadResult = roadModel(frame, verbose=False)

            if roadResult[0].masks is not None:
                masks = roadResult[0].masks.xy
                scores = roadResult[0].boxes.conf

                maxScore = np.argmax(scores)
                mask = np.array(masks[maxScore], dtype=np.int32).reshape(-1, 1, 2)

                maskImg = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillPoly(maskImg, [mask], (0, 255, 0))

                alpha = 0.25

                trafficLightResult = trafficLightModel(frame, verbose=False, conf=0.6)

                if trafficLightResult and len(trafficLightResult[0].boxes) > 0:
                    for result in trafficLightResult[0].boxes:
                        x1, y1, x2, y2 = map(int, result.xyxy[0])
                        confidence = math.ceil((result.conf[0] * 100)) / 100
                        cls = int(result.cls[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{classNames[cls]} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if classNames[cls] == "green":
                            frame = detect(frame, yoloModel, maskImg, "green")
                        elif classNames[cls] == "yellow" or classNames[cls] == "red":
                            frame = detect(frame, yoloModel, maskImg, "notGreen")
                else:
                    frame = detect(frame, yoloModel, maskImg, None)

                frame = cv2.addWeighted(frame, 1, maskImg, alpha, 0)

                edges = cv2.Canny(maskImg, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cv2.drawContours(frame, contours, -1, lineColor, 2)

            out.write(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing completed. Output saved at {outputVideoPath}")


def parseOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/videoRoad1.mp4', help='file/dir')
    parser.add_argument('--outputVideoPath', type=str, default='output_video.mp4', help='Output video file path')
    parser.add_argument('--yoloModelSource', type=str, default="data/yolo11s.pt", help='Path to YOLO model file')
    parser.add_argument('--roadModelSource', type=str, default='data/road/best.pt', help='Path to road segmentation model file')
    parser.add_argument('--trafficLightModelSource', type=str, default='data/trafficLight.pt', help='Path to traffic light detection model file')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parseOpt()
    argsRun(**vars(opt))
