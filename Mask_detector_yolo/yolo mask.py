# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 12:24:32 2021

@author: DST project
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import glob
from tqdm import tqdm

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# model = load_model('gender_detection2.model')
# classes = ['men','woman']
# classes = ['person'] 
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
# load the face mask detector model from disk
maskNet = load_model("emotion_detection_model_100epochs.h5")

with open("coco.names", "r") as f:
    classess = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image

# image_files = [f for f in glob.glob(r'E:\Gender-Detection-master\images' + "/**/*", recursive=True) if not os.path.isdir(f)]
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
def rescaleFrame(frame, scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width, height)
    return cv2.resize (frame,dimensions,interpolation=cv2.INTER_AREA)

def detect_and_predict_mask(face_crop, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame_resized.shape[:2]
	blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
# 	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			face = cv2.resize(face, (48, 48))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
      
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
count=0   
cap = cv2.VideoCapture(0) 
while True:
    _, frame = cap.read()
    # for path in tqdm(image_files, total=len(image_files)):    
       
    # z = cv2.imread(frame, cv2.IMREAD_COLOR) 
    frame_resized=rescaleFrame(frame)
    frame_id += 1
  
    height, width, channels = frame_resized.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            
            class_id = np.argmax(scores)
            
            confidence = scores[class_id]
            
            if float(confidence) > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
          
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    
   
    for i in range(0,len(boxes)):
       
       if i in indexes:
    
            x, y, w, h = boxes[i]
            # print(boxes[1])
           
            label = str(classess[class_ids[i]])
            
            if label == 'person':
            
                confidence = confidences[i]
                
                # color = colors[class_ids[i]]
                
                # cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
                
                face_crop = np.copy(frame_resized[y:y + h,x:x + w])
        
                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue
        
                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (224,224))
                face_crop = face_crop.astype("float") / 255.0
                
                face_crop = img_to_array (face_crop)
                # print(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
               
                (locs, preds) = detect_and_predict_mask(face_crop, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
            for (box, pred) in zip(locs, preds):
          		# unpack the bounding box and predictions
                  (startX, startY, endX, endY) = box
          	      
                  (mask, withoutMask, aa, bb, cc, dd, ee) = pred
                  predictions = np.argmax(pred)
                  print(predictions)
                  ff=class_labels[predictions]
          		# determine the class label and color we'll use to draw
          		# the bounding box and tex
                  # label = "Mask" if mask > withoutMask else "No Mask"
          	      
                  color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
          
          		# include the probability in the label
                  labell = "{}: {:.2f}%".format(ff, max(mask, withoutMask,aa,bb,cc,dd,ee) * 100)
          
          		# display the label and bounding box rectangle on the output
          		# frame
                  cv2.putText(frame_resized, labell, (startX, startY - 10),
          			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                  cv2.rectangle(frame_resized, (startX, startY), (endX, endY), color, 2)
                  cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # show the output frame
    cv2.imshow("Frame", frame_resized)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()