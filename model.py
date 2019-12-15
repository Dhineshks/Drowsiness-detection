import os
import time
import cv2
from cv2 import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer

path = os.getcwd()

#initialization for alert sound
mixer.init()

alert = mixer.Sound('/sounds/alert1.wav')
 
#note :- HaarCascade is trained by superimposing the positive images over a set of negative images
#haar cascade classifier for face
face = cv2.CascadeClassifier('/HaarCascadeFiles/haarcascade_frontalface_alt.xml')
#haar cascade classifier for left eye
lefteye = cv2.CascadeClassifier('/HaarCascadeFiles/haarcascade_lefteye_2splits.xml')
#haar cascade classifier for right eye
righteye = cv2.CascadeClassifier('/HaarCascadeFiles/haarcascade_righteye_2splits.xml')

eye = ['Closed','Open']

#load CNN model
model = load_model('/cnnmodel/cnn.h5')