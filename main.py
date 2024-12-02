# import opencv library
import cv2

# OpenCV DNN
# initialize Neural Network
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
# define a model
model = cv2.dnn.DetectionModel()
model.setInputParams(size=(320, 320), )

# initialize Web Camera
# 0 takes the first camera, 1 - the second, 2 - the third, etc.
cap = cv2.VideoCapture(0)

# we don't need only one frame (image), but we need the video
# which is why we have the loop
while True:
    # ret is saying if we have the frame (True) or not (False)
    ret, frame = cap.read()
    # show the frame
    cv2.imshow("Frame", frame)
    # wait until the key is pressed to keep the frame open until told otherwise
    # we can't have 0 here -> this will mean that the program will change frames only
    # if the key was pressed
    cv2.waitKey(1)
