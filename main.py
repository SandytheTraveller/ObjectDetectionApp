# import opencv library
import cv2
import numpy as np

# OpenCV DNN
# --> Initialize Neural Network
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
# define a model + pass a network to the model
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255) # bigger size means better precision on detection, but slower

# --> Load classes' list
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# --> Initialize Web Camera
# 0 takes the first camera, 1 - the second, 2 - the third, etc.
cap = cv2.VideoCapture(0)
# change the camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Full HD: 1920 x 1080


def click_button(event, x, y, flags, params):
    # flag which tells if the button is activated
    global button_person

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
        is_inside_polygon = cv2.pointPolygonTest(polygon, (x, y), False)
        if is_inside_polygon > 0:
            print("We are clicking inside the button")
            if not button_person:
                button_person = True
            else:
                button_person = False


# --> Create window to associate mouse with it
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

# we don't need only one frame (image), but we need the video
# which is why we have the loop
while True:
    # ret is saying if we have the frame (True) or not (False)
    ret, frame = cap.read()

    # --> Object Detection
    (class_ids, scores, bboxes) = model.detect(frame) # we get classes, scores and bounding boxes
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, width, height) = bbox # coordinates - parameters - of the bounding box
        class_name = classes[class_id]

        # the bounding box is appearing only for class Person
        if class_name == 'person':
            cv2.putText(frame, str(class_name), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 50), 2) # y-5, so that it's with the object but doesn't overlap with it
            # draw bounding boxes
            cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 0, 50), 3) # where: frame; starting point; ending point; colour; thickness

    '''print("Class ids: ", class_ids)
    print("Scores: ", scores)
    print("BBoxes: ", bboxes)'''
    # --> Create Button
    # cv2.rectangle(frame, (20, 20), (220, 70), (0, 200, 0), -1) # thickness -1: rectangle is completely filled with the colour
    # we check if we clicked the button, i.e.e we check the coordinates of the click
    polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
    cv2.fillPoly(frame, polygon, (0, 200, 0))
    cv2.putText(frame, "Person", (35, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)


    # show the frame
    cv2.imshow("Frame", frame)
    # wait until the key is pressed to keep the frame open until told otherwise
    # we can't have 0 here -> this will mean that the program will change frames only
    # if the key was pressed
    cv2.waitKey(1)
