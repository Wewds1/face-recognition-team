import cv2
import numpy as np
import mtcnn
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
from keras_facenet import FaceNet
from config import FACELOADING
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

embedder = FaceNet()
encoder = LabelEncoder()

# # Load image
# img_path = 'faces/allen/3.jpg'
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Detect faces
# results = detector.detect_faces(img)

# x, y, w, h = results[0]['box']
# img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# the_face = img[y:y + h, x:x + w]

# print(the_face)





# Test the class
print("Creating FACELOADING instance...")
faceloading = FACELOADING('faces')

print("Loading classes...")
X, Y = faceloading.load_classes()



def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

EMBEDDED_X = []

for image in X:
    EMBEDDED_X.append(get_embedding(image))

EMBEDDED_X = np.asarray(EMBEDDED_X)



encoder.fit(Y)
encoded_y = encoder.transform(Y)


X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, encoded_y, test_size=0.2, random_state=42, stratify=encoded_y)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

np.savez_compressed('embeddings.npz', embeddings=EMBEDDED_X, labels=Y, encoded_labels=encoded_y)


with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoder': encoder, 'accuracy': accuracy}, f)


for i in range(min(20, len(X_test))):
    prediction = model.predict([X_test[i]])[0]
    probability = model.predict_proba([X_test[i]])[0]
    predicted_name = encoder.inverse_transform([prediction])[0]
    actual_name = encoder.inverse_transform([Y_test[i]])[0]
    confidence = max(probability)
    
    print(f"Sample {i+1}: Predicted '{predicted_name}' (confidence: {confidence:.2f}) | Actual: '{actual_name}'")
faceloading.plot_images()
