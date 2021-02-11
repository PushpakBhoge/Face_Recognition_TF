from scipy.spatial import distance
from Inference.inception_resnet_v1 import InceptionResNetV1
import tensorflow as tf
import numpy as np
import json
import os
import cv2 

class FaceRecognizer():
    def __init__(self, database_path=None):
        """
		Arguments:
        database_path - path to json file holding embeddings of faces with name
          format - {"Name1":[embedding1],
                    "Name2":[embeddin2]}
        Generating the json file using Load_people_into_DataBase is recommended

		"""
        # get current directory
        self.cwdir = os.path.curdir
        # Set base directory for converted weights
        self.WEIGHT_BASE = os.path.join('Model','model_weights')
        # Check if keras saved weights exists load from them if exists
        # if any error then load from the extracted weights
        try:
            model_path = os.path.join(self.cwdir, 'Model', 'FaceNet_Keras_converted.h5')
            self.model = tf.keras.models.load_model(model_path)
        except:
            self.model = self.load_model()
        # Load DataBase
        if database_path==None:
            database_path = os.path.join(self.cwdir, 'DataBase', 'DataBase.json')
        with open(database_path, "r") as file:
            self.database = json.load(file)
    
    def load_model(self):
        """
		Arguments:
        None
		Output:
        a model instance
        This method is not meant to be called outside class
		"""
        # load model from source
        model = InceptionResNetV1()

        # Load weights layer by layer
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
    
    # method to export model
    def export_model(self, path=None):
        """
		Arguments:
        path - output path of the model include .h5 at the end to save as keras
                model else provide directory if want saved_model format
		Output:

		"""
        if path == None:
            path = os.path.join("Model", "FaceNet_Keras_converted.h5")
        self.model.save(path)
    
    # method that preprocess iamges
    def preprocess_image(self, face_img):
        """
		Arguments:
        face_img = Face crop of the image
		Output:
        return preprocessed(normalized) version of image

		"""
        # resize image and converty to recommended data type
        img = cv2.resize(face_img, (160,160))
        img = np.asarray(img, 'float32')

        axis = (0,1,2)
        size = img.size

        mean = np.mean(img, axis=axis, keepdims=True)
        std = np.std(img, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        processed_img = (img-mean) / std_adj

        return processed_img

    # l2 normalize embeddindgs
    def l2_normalize(self, embed, axis=-1, epsilon=1e-10):
        """
		Arguments:
        embed - 128 number long embeddind 
        axis - axis of the embedding default to -1
        epsilon - a small number to avoid division by zero 
		Output:
        normalized version of embeddings

		"""
        output = embed / np.sqrt(np.maximum(np.sum(np.square(embed), axis=axis, keepdims=True), epsilon))
        return output
    
    # method for getting face embeddings using model 
    def get_face_embedding(self, face):
        """
		Arguments:
        face - face crop drom an image
		Output:
        face embedding with 128 parameters

		"""
        # preprocess iamge and expand the dimension 
        processed_face = self.preprocess_image(face)
        processed_face = np.expand_dims(processed_face, axis=0)

        # predict using model and l2 normalize embedding
        model_pred = self.model.predict(processed_face)
        face_embedding = self.l2_normalize(model_pred)
        return face_embedding
    
    # calculate euclidain distance between the true and predicted 
    # face embeddings
    def calculate_distance(self, embd_real, embd_candidate):
        """
		Arguments:
        embd_embd - embedding from database
        embd_candidate - model predicted embedding
		Output:
        euclidian distance between the two embeddings

		"""
        return distance.euclidean(embd_real, embd_candidate)
    
    # Function whch compare predicted embedding face with embedding in database
    # and return result with least distance as a person name
    # if minimum distance is greater than 1 then person name is printed 
    # as UNKNOWN person
    def Whoisit(self, face_embedding):
        """
		Arguments:
        face_embeddings - face embedding vector of 128 dimension predicted 
                            by model
		Output: tuple
        person_name - Name of the person from database where distance is minimum
        minimum_distance - scaler of the distance from the detected person
		"""
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
        return person_name, minimum_distance