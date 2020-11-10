import face_recognition
import os
import cv2
import pickle
KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.47
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
print('Loading known faces...')
known_faces = []
known_names = []
for name in os.listdir(KNOWN_FACES_DIR):
    print("name = ", name)
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print("filename = ", filename)
        #ima = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        #image = image_resize(ima, height = 800)
        #encoding = face_recognition.face_encodings(image)
        with open(f"{KNOWN_FACES_DIR}/{name}/{filename}", 'rb') as f:
            encoding = pickle.load(f)
        known_faces.append(encoding)
        known_names.append(name)

        with open('dataset_faces_test.dat', 'wb') as f:
            pickle.dump(known_faces,f)
        with open('dataset_names_test.dat', 'wb') as f:
            pickle.dump(known_names,f)