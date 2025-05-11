import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

CASCADE_DIR = "models/cascade"

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml"))
        self.mouth_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_mcs_mouth.xml"))

    def detect_faces(self, img):
        face_img = img.copy()
        face_rect = self.face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in face_rect:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
            
        return face_img, face_rect

    def detect_mouth(self, img):
        mouth_img = img.copy()
        mouth_rect = self.mouth_cascade.detectMultiScale(mouth_img, scaleFactor=1.5, minNeighbors=11)

        for (x, y, w, h) in mouth_rect:
            cv2.rectangle(mouth_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

        return mouth_img, mouth_rect

    def identify_face_and_mouth(self, image_path):
        image = cv2.imread(image_path)
        face_img, face_rect = self.detect_faces(image)

        plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite('data/face.jpg', face_img)

        faces = []
        for face in face_rect:
            face_img = image[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            faces.append({"face": face_img, "rect": face})

        for face in faces:
            mouth_img, mouth_rect = self.detect_mouth(face["face"])
            face["mouth"] = mouth_img

        for face in faces:
            plt.imshow(cv2.cvtColor(face["mouth"], cv2.COLOR_BGR2RGB))
            plt.show()
            cv2.imwrite(f'data/mouth_{face["rect"][0]}.jpg', face["mouth"])

if __name__ == "__main__":
    detector = FaceDetector()
    detector.identify_face_and_mouth("data/matt.jpg")