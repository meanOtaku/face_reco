import face_recognition
import os
import cv2
import pickle
#KNOWN_FACES_DIR = 'known_faces'
#UNKNOWN_FACES_DIR = 'unknown_faces'
video = cv2.VideoCapture(0)
TOLERANCE = 0.47
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  
known_faces = []
known_names = []
with open('dataset_faces.dat','rb') as f:
    known_faces = pickle.load(f)
with open('dataset_names.dat','rb') as f:
    known_names = pickle.load(f)

def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
print('Loading video...')

name = input("Enter The Name : ")
while True:
    ret, image = video.read()
    ima = face_recognition.load_image_file(image)
    locations = face_recognition.face_locations(ima,  model='cnn')
    encoding = face_recognition.face_encodings(ima)
    known_faces.append(encoding)
    known_names.append(name)

    with open('dataset_faces_video.dat', 'wb') as f:
        pickle.dump(known_faces,f)
    with open('dataset_names_video.dat', 'wb') as f:
        pickle.dump(known_names,f)

    for face_encoding, face_location in zip(encoding, locations):
            for i in range(0,len(known_faces)):  
                results = face_recognition.compare_faces(known_faces[i], face_encoding, TOLERANCE)
                match = "Recording"
                color = 0
                

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                


                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow("filename", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break