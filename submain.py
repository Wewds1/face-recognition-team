import cv2
import numpy as np
import mtcnn
import pickle
import time
from keras_facenet import FaceNet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cap = cv2.VideoCapture('cam2.mp4')
embedder = FaceNet()
detector = mtcnn.MTCNN()

frame_number = 0
intervals = {}
current_names = {}
missing_counts = {}
missing_tolerance = 5
fps = cap.get(cv2.CAP_PROP_FPS)

try:
    with open('face_recognition_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        encoder = saved_data['encoder']
        print(f"Model loaded! Training accuracy: {saved_data['accuracy']:.2%}")
except:
    print("No trained model found! Run the trainer first.")
    exit()


def secs_to_time(secs):
    m = int(secs//60)
    s = int(secs % 60)
    return f"{m:02d}:{s:02d}"

def merged_intervals(intervals, gap=1.0):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] > gap:
            merged.append((start, end))
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


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


while True:
    ret, frame = cap.read()
    if not ret:
        break 
    frame_number += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)
    names_in_frame = set()
    for result in results:
        x, y, w, h = result['box']
        confidence_detection = result['confidence']
    
        if confidence_detection > 0.8:
            face = rgb_frame[y:y+h, x:x+w]
            name, confidence_recognition = recognize_face(face)
            names_in_frame.add(name)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence_recognition:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    for name in names_in_frame:
        if name not in current_names:
            current_names[name] = frame_number
        missing_counts[name] = 0

    for name in list(current_names.keys()):
        if name not in names_in_frame:
            missing_counts[name] = missing_counts.get(name, 0) + 1
            if missing_counts[name] >= missing_tolerance:
                start_frame = current_names[name]
                end_frame = frame_number 
                start_time = start_frame / fps
                end_time = end_frame / fps

                if name not in intervals:
                    intervals[name] = []
                intervals[name].append((start_time, end_time))
                del current_names[name]
                del missing_counts[name]
        else:
            missing_counts[name] = 0

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for name, start_frame in current_names.items():
    end_frame = frame_number
    start_time = start_frame / fps
    end_time = end_frame / fps
    if name not in intervals:
        intervals[name]= []
    intervals[name].append((start_time, end_time))

cap.release()
cv2.destroyAllWindows()

print("Face Recognition Results:")
for name, times in intervals.items():
    merged_times = merged_intervals(times)
    out = []
    for start, end in merged_times:
        out.append(f"{secs_to_time(start)} - {secs_to_time(end)}")
    print(f"{name}: {', '.join(out)}")