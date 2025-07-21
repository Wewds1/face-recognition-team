import cv2
import numpy as np
import mtcnn
import pickle
from keras_facenet import FaceNet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    with open('face_recognition_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        encoder = saved_data['encoder']
        print(f"Model loaded! Training accuracy: {saved_data['accuracy']:.2%}")
except:
    print("No trained model found! Run the trainer first.")
    exit()

embedder = FaceNet()
detector = mtcnn.MTCNN()

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

def recognize_face(face_img, threshold=0.7):
    try:
        embedding = get_embedding(face_img)
        prediction = model.predict([embedding])[0]
        probability = model.predict_proba([embedding])[0]
        confidence = max(probability)
        
        if confidence > threshold:
            name = encoder.inverse_transform([prediction])[0]
            return name, confidence
        else:
            return "Unknown", confidence
    except:
        return "Error", 0.0

cap = cv2.VideoCapture('rtsp://admin:c0smeti123@10.10.10.98:554/cam/realmonitor?channel=8&subtype=0&unicast=true&proto=Onvif')  # Use webcam



while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)
    
    for result in results:
        x, y, w, h = result['box']
        confidence_detection = result['confidence']
        
        if confidence_detection > 0.8:  # Only process high-confidence detections
            face = rgb_frame[y:y+h, x:x+w]
            
            name, confidence_recognition = recognize_face(face)
            
            # Draw results
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence_recognition:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()