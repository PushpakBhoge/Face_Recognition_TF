from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import json
import os
import cv2 

class FaceRecognizer():
    def __init__(self):
        self.cwdir = os.path.curdir
        self.MODEL_JSON = os.path.join(self.cwdir,'Model','FaceNet_Architecture.json')
        self.WEIGHT_BASE = os.path.join('Model','model_weights')
        if "FaceNet_Keras_converted.h5" not in os.listdir(os.path.join(self.cwdir, 'Model')):
            self.model = self.load_model()
        else:
            model_path = os.path.join(self.cwdir, 'Model', 'FaceNet_Keras_converted.h5')
            self.model = tf.keras.models.load_model(model_path)
        database_path = os.path.join(self.cwdir, 'DataBase', 'DataBase.json')
        with open(database_path, "r") as file:
            self.database = json.load(file)
    
    def load_model(self):
        with open(self.MODEL_JSON, 'r') as file:
            model_architecturect = json.load(file)
            model = tf.keras.models.model_from_json(model_architecturect)

        layer_files = os.listdir(self.WEIGHT_BASE)
        for i, layer in enumerate(model.layers):
            weight_files = [x for x in layer_files if x.split(".")[0]==layer.name]
            for weight_file in weight_files:
                files_loaded = np.load(os.path.join(self.WEIGHT_BASE, weight_file))
                weights_for_layer = []
                for file in files_loaded:
                    weights_for_layer.append(files_loaded[file])
            try:
                layer.set_weights(weights_for_layer)
            except:
                pass

        return model
    
    def export_model(self, path=None):
        if path == None:
            path = os.path.join("Model", "FaceNet_Keras_converted.h5")
        self.model.save(path)
    
    def preprocess_image(self, face_img):
        img = cv2.resize(face_img, (160,160))
        img = np.asarray(img, 'float32')

        axis = (0,1,2)
        size = img.size

        mean = np.mean(img, axis=axis, keepdims=True)
        std = np.std(img, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        processed_img = (img-mean) / std_adj

        return processed_img

    def l2_normalize(self, embed, axis=-1, epsilon=1e-10):
        output = embed / np.sqrt(np.maximum(np.sum(np.square(embed), axis=axis, keepdims=True), epsilon))
        return output

    def get_face_embedding(self, face):
        processed_face = self.preprocess_image(face)
        processed_face = np.expand_dims(processed_face, axis=0)

        model_pred = self.model.predict(processed_face)
        face_embedding = self.l2_normalize(model_pred)
        return face_embedding
    
    def calculate_distance(self, embd_real, embd_candidate):
        return distance.euclidean(embd_real, embd_candidate)
    
    def Whoisit(self, face_embedding):
        distance = {}
        minimum_distance = None
        person_name = ""
        for name, embedding in self.database.items():
            distance[name] = self.calculate_distance(embedding, face_embedding)
            if minimum_distance == None or distance[name]<minimum_distance:
                minimum_distance = distance[name]
                person_name = name
        if minimum_distance>1:
            person_name = "UNKNOWN"
            minimum_distance = 2
        return person_name, minimum_distance