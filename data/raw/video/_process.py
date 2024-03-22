import cv2
import os
import sys

WD = os.path.dirname(__file__)
CROPPED = os.path.join(WD, "cropped")
FRAMES = os.path.join(WD, "frames")

for image in os.listdir(FRAMES):
    img = cv2.imread(os.path.join(FRAMES, image))

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(os.path.join(WD, 'haarcascade_frontalface_alt2.xml'))

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]

        cv2.imwrite(os.path.join(WD, "cropped", image), faces)