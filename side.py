from ultralytics import YOLO
import numpy as np
import math
import cv2

def cornerInMask(mask, x1, y1, x2, y2):
    # Nesne köşelerini belirle
    corners = [
        (x1, y1),  # Sol üst
        (x2, y1),  # Sağ üst
        (x1, y2),  # Sol alt
        (x2, y2),  # Sağ alt
    ]

    inMask = False
    
    # Her köşeyi kontrol et
    for (x, y) in corners:
        if mask[y, x, 1] == 255:  # Yeşil kanalı kontrol et
            inMask = True
            break
     
    return inMask

def detect(frame, model, maskImg, trafficLightState):
    height, width = frame.shape[:2]

    rightLaneStatus = "empty"
    middleLaneStatus = "empty"
    leftLaneStatus = "empty"

    detectionCoefficient = 1
    detectionCount = 0

    dangerousObjectDistance = None
    driveConf = 1
    drivingDecision = "There is no problem for driving. Have a nice ride"

    notDangerousObjectColor = (255, 0, 0)
    dangerousObjectColor = (0, 0, 255)
    textColor = (0, 0, 0)

    # X ve Y çizgilerini hesaplayıp listelere atama
    lineX = [(i * (width // 17)) for i in range(1, 18)]
    lineY = [(i * (height // 17)) for i in range(1, 18)]

    if trafficLightState in [None, "green"]:
        cocoData = [0, 1, 2, 3, 5, 7, 9]

        modelResult = model(frame, verbose=False, conf=0.65)

        trafficLightStateText = f"Detect Color: {trafficLightState}. Have a nice drive!"  

        for result in modelResult:
            for box in result.boxes:

                if (detectionCount != 0):
                    detectionCoefficient -= 0.02

                if box.cls in cocoData:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()

                    detectionCount += 1

                    match box.cls:
                        case 0:
                            label = f"Person {conf:.2f}"
                            stateName = "person"
                        case 1:
                            label = f"Bicycle {conf:.2f}"
                            stateName = "bicycle"
                        case 2:
                            label = f"Car {conf:.2f}"
                            stateName = "car"
                        case 3:
                            label = f"Motorcycle {conf:.2f}"
                            stateName = "motorcycle"
                        case 5:
                            label = f"Bus {conf:.2f}"
                            stateName = "bus"
                        case 7:
                            label = f"Truck {conf:.2f}"
                            stateName = "truck"

                    detectionObjectsPositionX = int((x1 + x2) // 2)
                    detectionObjectsPositionY = int((y1 + y2) // 2)

                    hipo = 0                                              

                    if (cornerInMask(maskImg, x1, y1, x2, y2)):

                        # Yükseklik sınırı kontrol etme..
                        if (detectionObjectsPositionY >= lineY[5]):

                            camPosX = lineX[13]
                            camPosY = (lineY[7] + lineY[8]) // 2                            

                            # Sol şerit
                            if (
                                detectionObjectsPositionX <= lineX[1] or
                                (lineX[1] <= detectionObjectsPositionX <= lineX[2] and detectionObjectsPositionY <= lineY[13]) or
                                (lineX[2] <= detectionObjectsPositionX <= lineX[3] and detectionObjectsPositionY <= lineY[12]) or 
                                (lineX[3] <= detectionObjectsPositionX <= lineX[4] and detectionObjectsPositionY <= lineY[11]) or 
                                (lineX[4] <= detectionObjectsPositionX <= lineX[5] and detectionObjectsPositionY <= lineY[10]) or 
                                (lineX[5] <= detectionObjectsPositionX <= lineX[6] and detectionObjectsPositionY <= lineY[9])
                            ):
                                hipo = math.sqrt((camPosX - int((x1 + x2) // 2))**2 + (int(((y1 + y2) // 2)) - camPosY)**2)
                                if (hipo >= 435):
                                    dangerousObjectDistance = "far"
                                elif (hipo >= 310):
                                    dangerousObjectDistance = "middle"
                                else :
                                    dangerousObjectDistance = "near"

                                print("sol", hipo)
                                leftLaneStatus = f"{stateName} detection in left lane. Distance: {dangerousObjectDistance}"

                            # Orta Şerit
                            elif (
                                lineX[7] <= detectionObjectsPositionX <= lineX[8] or
                                ((lineX[6] <= detectionObjectsPositionX <= lineX[7] or lineX[8] <= detectionObjectsPositionX <= lineX[9]) and detectionObjectsPositionY >= lineY[5]) or
                                ((lineX[5] <= detectionObjectsPositionX <= lineX[6] or lineX[9] <= detectionObjectsPositionX <= lineX[10]) and detectionObjectsPositionY >= lineY[9]) or
                                ((lineX[4] <= detectionObjectsPositionX <= lineX[5] or lineX[10] <= detectionObjectsPositionX <= lineX[11]) and detectionObjectsPositionY >= lineY[10]) or
                                ((lineX[3] <= detectionObjectsPositionX <= lineX[4] or lineX[11] <= detectionObjectsPositionX <= lineX[12]) and detectionObjectsPositionY >= lineY[11]) or
                                ((lineX[2] <= detectionObjectsPositionX <= lineX[3] or lineX[12] <= detectionObjectsPositionX <= lineX[13]) and detectionObjectsPositionY >= lineY[12]) or
                                ((lineX[1] <= detectionObjectsPositionX <= lineX[2] or lineX[13] <= detectionObjectsPositionX <= lineX[14]) and detectionObjectsPositionY >= lineY[13])
                            ):
                                
                                hipo = math.sqrt((int((x1 + x2) // 2) - camPosX)**2 + (int(((y1 + y2) // 2)) - camPosY)**2)
                                if (hipo >= 475):
                                    dangerousObjectDistance = "far"
                                elif (hipo >= 310):
                                    dangerousObjectDistance = "middle"
                                else :
                                    dangerousObjectDistance = "near"

                                print("orta", hipo)
                                middleLaneStatus = f"{stateName} detection in middle lane. Distance: {dangerousObjectDistance}"
                            
                            # Sağ şerit
                            elif (
                                detectionObjectsPositionX >= lineX[14] or
                                (lineX[13] <= detectionObjectsPositionX <= lineX[14] and detectionObjectsPositionY <= lineY[13]) or
                                (lineX[12] <= detectionObjectsPositionX <= lineX[13] and detectionObjectsPositionY <= lineY[12]) or 
                                (lineX[11] <= detectionObjectsPositionX <= lineX[12] and detectionObjectsPositionY <= lineY[11]) or 
                                (lineX[10] <= detectionObjectsPositionX <= lineX[11] and detectionObjectsPositionY <= lineY[10]) or 
                                (lineX[9] <= detectionObjectsPositionX <= lineX[10] and detectionObjectsPositionY <= lineY[9])
                            ):
                                
                                hipo = math.sqrt((int((x1 + x2) // 2) - camPosX)**2 + (int(((y1 + y2) // 2)) - camPosY)**2)

                                if (hipo >= 435):
                                    dangerousObjectDistance = "far"
                                elif (hipo >= 320):
                                    dangerousObjectDistance = "middle"
                                else :
                                    dangerousObjectDistance = "near"

                                print("sag", hipo)
                                rightLaneStatus = f"{stateName} detection in right lane. Distance: {dangerousObjectDistance}"
                                                        
                        objectsColor = dangerousObjectColor
                    else :
                        objectsColor = notDangerousObjectColor                        


                    # Detect object control point
                    cv2.circle(frame, (detectionObjectsPositionX, detectionObjectsPositionY), 7, objectsColor, -1)
                    if (hipo != 0):
                        cv2.putText (frame, f"{round(hipo, 4)}" ,(detectionObjectsPositionX, detectionObjectsPositionY -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, objectsColor, 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), objectsColor, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, objectsColor, 2)

                    if (trafficLightState == None): driveConf *= 0.98
                    elif (trafficLightState == "green"): driveConf *= 0.95

                    if (dangerousObjectDistance == "far"): driveConf *= (0.94 * detectionCoefficient)
                    elif (dangerousObjectDistance == "middle"): driveConf *= (0.87 * detectionCoefficient)
                    elif (dangerousObjectDistance == "near"): driveConf *= (0.82 * detectionCoefficient)

                    if (cornerInMask(maskImg, x1, y1, x2, y2)):
                        if (stateName == "person"): driveConf *= (0.93 * detectionCoefficient)
                        elif (stateName == "car" or stateName == "bus"): driveConf *= (0.85 * detectionCoefficient)
                        elif (stateName == "truck"): driveConf *= (0.76 * detectionCoefficient)
                        elif (stateName == "bicycle" or stateName == "motorcycle"): driveConf *= (0.87 * detectionCoefficient)

                        
                    if (trafficLightState == "notGreen"):
                        driveConf *= 0

    else :        
        trafficLightStateText = f"Detect Color: {trafficLightState}."        

    """
    for x in lineX:
        cv2.line(frame, (x, 0), (x, height), (0, 0, 0), 2)

    for y in lineY:
        cv2.line(frame, (0, y), (width, y), (0, 0, 0), 2)
    """

    driveConf = round(driveConf, 2)
    print(f"Güven değeri: {driveConf}")


    if (driveConf >= 0.9):
        drivingDecision = "There is no problem for driving. Have a nice ride"
    elif (driveConf >= 0.75):
        drivingDecision = "Do not neglect environmental controls. Have a nice drive"
    elif (driveConf >= 0.60):
        drivingDecision = "Environmental situations should be reviewed again. be careful"
    elif (driveConf >= 0.35):
        drivingDecision = "Possible emergency. Slow down your speed in a controlled manner"
    elif (driveConf >= 0.15):
        drivingDecision = "Possible emergency. Slow down your speed immediately"
    else :
        drivingDecision = "Emergency. You stopped the car!!"

    # Metin ekleme
    cv2.putText(frame, f"Right Lane Status: {rightLaneStatus}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, textColor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Middle Lane Status: {middleLaneStatus}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, textColor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Left Lane Status: {leftLaneStatus}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, textColor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Traffic Light Status: {trafficLightStateText}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, textColor, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Driving decision: {drivingDecision}", (width//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, textColor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Driving confidence: {driveConf}", (width//2 - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, textColor, 2, cv2.LINE_AA)

    return frame

