import os
import time
import cv2
from cv2 import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer

path = os.getcwd()
#initializing mixer for alert sound
mixer.init()
alert = mixer.Sound('/home/dhinesh/machinelearning/github/Drowsiness-detection/sounds/alert2.wav')
# note :- HaarCascade is trained by superimposing the positive images over a set of negative images
# haar cascade classifier for face
face = cv2.CascadeClassifier('/home/dhinesh/machinelearning/github/Drowsiness-detection/HaarCascadeFiles/haarcascade_frontalface_alt.xml')
# haar cascade classifier for left eye
lefteye = cv2.CascadeClassifier('/home/dhinesh/machinelearning/github/Drowsiness-detection/HaarCascadeFiles/haarcascade_lefteye_2splits.xml')
# haar cascade classifier for right eye
righteye = cv2.CascadeClassifier('/home/dhinesh/machinelearning/github/Drowsiness-detection/HaarCascadeFiles/haarcascade_righteye_2splits.xml')

eye = ['Closed','Open']
#cnn model
model = load_model('/home/dhinesh/machinelearning/github/Drowsiness-detection/cnnmodel/cnn.h5')
#opens default webcam
capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0 
thres = 2
rightpred = [99]
leftpred = [99]

while(True):
	ret,frame = capture.read()
	height,width = frame.shape[:2]
	# frame = np.array(frame,dtype=np.uint8)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25),flags=cv2.CASCADE_SCALE_IMAGE)
	left_eye = lefteye.detectMultiScale(gray)
	right_eye = righteye.detectMultiScale(gray)
	cv2.rectangle(frame,(0,height - 50),(200,height),(0,0,0),thickness=cv2.FILLED)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
	for (x,y,w,h) in right_eye:
		r_eye = frame[y:y+h,x:x+w]
		count = count +1
		r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
		r_eye = cv2.resize(r_eye,(24,24))
		r_eye = r_eye/255
		r_eye = r_eye.reshape(24,24,-1)
		r_eye = np.expand_dims(r_eye,axis=0)
		rightpred = model.predict_classes(r_eye)
		if(rightpred[0]==1):
			eye = 'Open'
		if(rightpred[0]==0):
			eye = 'Closed'
		break
	for (x,y,w,h) in left_eye:
		l_eye = frame[y:y+h,x:x+w]
		count = count + 1
		l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
		l_eye = cv2.resize(l_eye,(24,24))
		l_eye = l_eye/255
		l_eye = l_eye.reshape(24,24,-1)
		l_eye = np.expand_dims(l_eye,axis=0)
		leftpred = model.predict_classes(l_eye)
		if(leftpred[0]==1):
			eye = 'Open'
		if(leftpred[0]==0):
			eye = 'Closed'
		break
	if(rightpred[0]==0 and leftpred[0]==0):
		score = score + 1
		cv2.putText(frame,"Closed",(10,height-20),font,1,(0,0,255),1,cv2.LINE_AA)
	else:
		score = score -1
		cv2.putText(frame,"Opened",(10,height-20),font,1,(0,255,0),1,cv2.LINE_AA)
	if(score<0):
		score=0
	cv2.putText(frame,"Score:"+str(score),(95,height-20),font,1,(255,255,255),1,cv2.LINE_AA)

	if(score>15):
		cv2.imwrite(os.path.join(path,'img.jpg'),frame)
		try:
			alert.play()
		except:
			pass
		if(thres<16):
			thres = thres + 2
		else:
			thres = thres - 2
			if(thres<2):
				thres = 2
		cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thres)
		cv2.putText(frame,"SLEEP ALERT!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
	
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
capture.release()
cv2.destroyAllWindows()