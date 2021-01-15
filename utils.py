from Inference.FaceDetector import FaceDetector
from Inference.FaceRecognizer import FaceRecognizer
import numpy as np
import cv2
import os

class Detector():
    def __init__(self):
        self.FaceDetect = FaceDetector()
        self.FaceRecog = FaceRecognizer()
    
    def get_people_names(self, image, speed_up=True, downscale_by=4):
        face_bboxes = self.FaceDetect.detect_faces(image, speed_up=speed_up, 
                                        scale_factor=downscale_by)
        Face_crops = self.FaceDetect.crop_faces(image, face_bboxes)

        results = []
        for face_crop, box in zip(Face_crops, face_bboxes):
            face_embd = self.FaceRecog.get_face_embedding(face_crop)
            person_name, distance = self.FaceRecog.Whoisit(face_embd)
            results.append((distance, person_name, box))
        
        return results

    def draw_results(self, image, infer_results, 
                    color=(255,0,0),box_thickness=None,
                    font_size=None, font_thickness=None, 
                    offset=None):
        img = image.copy()

        settings = self.get_draw_settings(image.shape)
        if offset == None:
            offset = settings[0]
        if font_size == None:
            font_size = settings[1]
        if font_thickness == None:
            font_thickness = settings[2]
        if box_thickness == None:
            box_thickness = settings[3]
        
        for result in infer_results:
            dist, name, box = result
            x1,y1,x2,y2 = box
            img = cv2.rectangle(img,(x1,y1),(x2,y2), 
                            color=color, thickness=box_thickness)
            text = "{} {:.2f}".format(name, dist)
            img = cv2.putText(img, text, (x1,y1-offset), 
                cv2.FONT_HERSHEY_SIMPLEX, font_size,
                color, font_thickness, cv2.LINE_AA)
        return img
    
    def get_draw_settings(self,image_shape):
        width,height,channels = image_shape
        offset = round(width/150)
        font_size = round(width/800, 2)
        font_thickness = round(width/400)
        box_thickness = round(width/300)
        return offset, font_size, font_thickness, box_thickness