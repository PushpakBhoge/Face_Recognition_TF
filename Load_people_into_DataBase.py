from Inference.FaceRecognizer import FaceRecognizer
import numpy as np
import cv2
import json
import os

PEOPLE_DIR = "People"

files = os.listdir("DataBase")
if "DataBase.json" not in files:
    with open(os.path.join("DataBase","DataBase.json"), "x") as file:
        empty_json = {}
        json.dump(empty_json,file)

def add_person_to_database(Name, feature_vector):
    # load DataBase.json File
    with open(os.path.join("DataBase", "DataBase.json"), "r") as file:
        data = json.load(file)

    # append the record
    data[Name] = feature_vector

    # Save updated record to DataBase.json
    with open(os.path.join("DataBase", "DataBase.json"), "w") as file:
        json.dump(data,file)
        print(f"{Name} is added to DataBase")

people = os.listdir(PEOPLE_DIR)
recog = FaceRecognizer()

for person in people:
    name = person.split('.')[0]
    face = cv2.imread(os.path.join(PEOPLE_DIR, person))
    face_embd = recog.get_face_embedding(face).flatten().tolist()
    add_person_to_database(name, face_embd)
