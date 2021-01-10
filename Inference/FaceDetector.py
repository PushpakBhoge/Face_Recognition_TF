from scipy.spatial import distance
from mtcnn.mtcnn import MTCNN
from time import time
import numpy as np
import json
import cv2

class FaceDetector():
	"""
	Arguments :
	model_type = 'resnet50', 'vgg16', 'senet50'
	"""
	def __init__(self):
		self.facedet = MTCNN()

	def detect_faces(self, img, speed_up=True, scale_factor=4):
		# Create a copy of org image
		org = img.copy()
		bboxes = []

		if speed_up:
			# calculate scale factor so that width 
			# will be approx 512 pixel
			img_shape = org.shape
			width = img_shape[0]
			height = img_shape[1]

			# Scale the image down to speed up
			# Yeah! It was weird for me too but seems if not reversed
			# the image get distorted
			pixels = cv2.resize(org, (height//scale_factor,width//scale_factor))

		else:
			pixels = org

		# predict the results
		results = self.facedet.detect_faces(pixels)

		# isolating bounding boxes from results
		for result in results:
			x,y,w,h = map(lambda co:0 if co<0 else co, result["box"])

			if w*h > 1000:
				if speed_up:
					bboxes.append([x*scale_factor,y*scale_factor, (x+w)*scale_factor, (y+h)*scale_factor])
				else:
					bboxes.append([x,y,x+w,y+h])
		return bboxes

	def draw_boxes(self, img, bboxes, color=(255,0,0), thickness=5):
		img_copy = img.copy()
		for box in bboxes:
			x1,y1,x2,y2 = box
			cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, thickness)
		return img_copy

	def crop_faces(self, img, bboxes):
		image = img.copy()
		face_crops = []
		for box in bboxes:
			x1,y1,x2,y2 = box
			face_crop = image[y1:y2, x1:x2]
			face_crops.append(face_crop)
		return face_crops
			