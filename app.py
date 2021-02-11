import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Inference.FaceDetector import FaceDetector
from Inference.FaceRecognizer import FaceRecognizer
from utils import Detector
import cv2
import numpy as np

# Set directories
BASE_DIR = "images"
list_imgs = os.listdir("images\\")

# create detector object
detector = Detector()

# loop over images
for im in list_imgs:
	# read image
	img = cv2.imread(os.path.join(BASE_DIR, im))
	# convert to RGB 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# get predictions and draw them on image
	predictions = detector.get_people_names(img,speed_up=False, downscale_by=1)
	annoted_image = detector.draw_results(img, predictions)

	# convert back to BGR (since using cv2)
	image = cv2.cvtColor(annoted_image, cv2.COLOR_RGB2BGR)
	# Save image and show the annoted iamge
	cv2.imwrite(f"Infered_image\\{im.split('.')[0]}_infered.png", image)
	cv2.imshow("image", image)
	cv2.waitKey(3)



