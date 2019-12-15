import os
import time
import cv2
from cv2 import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer

#initialization for alert sound
mixer.init()

alert = mixer.Sound('/sounds/alert1.wav')

#haar caccade classifier for face 
face = cv2.