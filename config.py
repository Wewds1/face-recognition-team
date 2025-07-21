import cv2
import matplotlib.pyplot as plt
import os 
import numpy as np 
import mtcnn

detector = mtcnn.MTCNN()


class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.size = (160, 160)
        self.Y = []
        self.X = []

    def extract_faces(self, filename):
        img = cv2.imread(filename)
        if img is None:
            print(f"Could not read image: {filename}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img)
        
        if not results:
            print(f"No faces detected in: {filename}")
            return None
        
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y + h, x:x + w]
        face_arr = cv2.resize(face, self.size)
        
        return face_arr

    def load_faces(self, dir):
        FACES = []
        if not os.path.exists(dir):
            print(f"Directory not found: {dir}")
            return FACES
        image_files = [f for f in os.listdir(dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No image files found in: {dir}")
            return FACES
        
        for im_name in image_files:
            try:
                path = os.path.join(dir, im_name)
                face = self.extract_faces(path)
                if face is not None:
                    FACES.append(face)
                    print(f"Loaded face from: {im_name}")
            except Exception as e:
               pass
        
        return FACES

    def load_classes(self):
        if not os.path.exists(self.directory):
            print(f"Main directory not found: {self.directory}")
            return np.array([]), np.array([])
        
        subdirs = [d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, d))]
        
        if not subdirs:
            print(f"No subdirectories found in: {self.directory}")
            return np.array([]), np.array([])
        
        for dir_name in subdirs:
            path = os.path.join(self.directory, dir_name)
            print(f"\nProcessing directory: {dir_name}")
            
            FACES = self.load_faces(path)
            if FACES:
                labels = [dir_name for _ in range(len(FACES))]
                self.X.extend(FACES)
                self.Y.extend(labels)
                print(f"Added {len(FACES)} faces for class '{dir_name}'")
            else:
                print(f"No faces loaded for class '{dir_name}'")

        if not self.X:
            print("No faces loaded from any directory!")
            return np.array([]), np.array([])
        
        return np.array(self.X), np.array(self.Y)

    def plot_images(self):
        if not self.X:
            pass
        
        num_images = min(20, len(self.X))
        ncols = 4
        nrows = (num_images + ncols - 1) // ncols
        
        plt.figure(figsize=(15, 4 * nrows))
        
        for num in range(num_images):
            plt.subplot(nrows, ncols, num + 1)
            plt.imshow(self.X[num])
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
