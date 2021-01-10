import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Inference.FaceDetector import FaceDetector
from Inference.FaceRecognizer import FaceRecognizer
from utils import Detector
import cv2
import numpy as np


BASE_DIR = "images"
list_imgs = os.listdir("images\\")

#  utils face recognition Test
detector = Detector()
for im in list_imgs:
	img = cv2.imread(os.path.join(BASE_DIR, im))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	predictions = detector.get_people_names(img,speed_up=False, downscale_by=1)
	annoted_image = detector.draw_results(img, predictions, 
					                    color=(255,0,0),box_thickness=2,
					                    font_size=1, font_thickness=1)

	image = cv2.cvtColor(annoted_image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(f"Infered_image\\{im}_infered.jpeg", image)
	cv2.imshow("image", image)
	cv2.waitKey(1)



"""# Face cropping test
for im in list_imgs:
	img = cv2.imread(os.path.join(BASE_DIR, im))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	face_detect = FaceDetector()
	face_recognizer = FaceRecognizer()

	BBoxes = face_detect.detect_faces(img, speed_up=True, scale_factor=4)
	faces = face_detect.crop_faces(img, BBoxes)

	image = face_detect.draw_boxes(img, BBoxes)

	for face, box in zip(faces, BBoxes):
		x1,y1,x2,y2 = box
		face_embd = face_recognizer.get_face_embedding(face)
		person_name, distance = face_recognizer.Whoisit(face_embd)
		face_recognizer.export_model()
		image = cv2.putText(image, person_name, 
					(x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX ,  
					4, (255, 0, 0), 13, cv2.LINE_AA) 
		
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(f"dump\\{im}_infered.jpeg", image)
	image_reshaped = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
	#cv2.imshow("image", image_reshaped)
	#cv2.waitKey(1)"""


"""# Read image
for im in list_imgs:
	img = cv2.imread(os.path.join(BASE_DIR, im))

	face_detect = Detector()
	BBoxes = face_detect.detect_faces(img, speed_up=True, scale_factor=16)

	image = face_detect.draw_boxes(img,BBoxes)
	image_reshaped = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))

	cv2.imshow("image.py", image_reshaped)
	#cv2.imwrite(f"image{im}", image)
	cv2.waitKey(1)"""

"""# Web Cam Trial
vid = cv2.VideoCapture(0) 
face_detect = Detector()
while(True):
	ret,frame = vid.read()
	BBoxes = face_detect.detect_faces(frame, speed_up=True, scale_factor=4)

	image = face_detect.draw_boxes(frame, BBoxes)

	cv2.imshow("Camera Feed", image)
	if cv2.waitKey(1)& 0xFF == ord('q'):
		break

vid.release()
cv2.dstroyAllWindows()"""



