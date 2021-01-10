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
                    color=(255,0,0),box_thickness=7,
                    font_size=3, font_thickness=10):
        img = image.copy()
        for result in infer_results:
            dist, name, box = result
            x1,y1,x2,y2 = box
            img = cv2.rectangle(img,(x1,y1),(x2,y2), 
                            color=color, thickness=box_thickness)
            text = "{} {:.2f}".format(name, dist)
            img = cv2.putText(img, text, (x1,y1-20), 
                cv2.FONT_HERSHEY_SIMPLEX, font_size,
                color, font_thickness, cv2.LINE_AA)
        return img