from Inference.FaceRecognizer import FaceRecognizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import json
import os

def train_classifier(data_dir, export_dir, batch_size=32):
    # create a Face Recognizer object
    faceNet = FaceRecognizer()

    # Create input pipeline
    datagen = ImageDataGenerator(preprocessing_function=faceNet.preprocess_image)
    train_gen = datagen.flow_from_directory(data_dir, target_size=(160,160),
                                batch_size=batch_size, shuffle=False)
    
    # Dictionary to decode labels
    class_decode = {v:k for (k,v) in train_gen.class_indices.items()}

    # Get true Labels
    face_labels = np.array(train_gen.classes)

    # Get face embeddings for all images
    face_embeddings = faceNet.model.predict(train_gen, steps=1+train_gen.samples//batch_size )

    # Split data into train-test
    trainX, testX, trainY, testY = train_test_split(face_embeddings, face_labels, 
                                                    test_size=0.2, random_state=28)

    # create svc classifier
    svc_clf = SVC(kernel='linear', probability=True, verbose=True)
    
    # train svc classifier
    svc_clf.fit(trainX, trainY)

    # predict on train and test set
    train_pred = svc_clf.predict(trainX)
    test_pred = svc_clf.predict(testX)

    # calculate train and test accuracies
    train_acc = accuracy_score(trainY, train_pred)
    test_acc = accuracy_score(testY,test_pred)

    # save the model and decode json
    with open(os.path.join(export_dir, "decode.json"), "w") as file:
        json.dump(class_decode, file)
    with open(os.path.join(export_dir, "svc_classifier.sav"), "wb") as file:
        joblib.dump(svc_clf, file)

    # print training accuracies
    print("training accuracy is",train_acc*100)
    print("testing accuracy is", test_acc*100)
    
    return svc_clf,class_decode