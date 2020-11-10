import face_recognition
import os
import cv2
import pickle
with open('dataset_faces.dat','rb') as f:
    known_faces = pickle.load(f)
with open('dataset_names.dat','rb') as f:
    known_names = pickle.load(f)

known_names.append("unknown")

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
print('Processing unknown faces...')
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(f'Filename {filename}', end='')
    ima = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    image = image_resize(ima, height = 800)
    locations = face_recognition.face_locations(image,  model='cnn')
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        for i in range(0,len(known_faces)):  
            results = face_recognition.compare_faces(known_faces[i], face_encoding, TOLERANCE)
            match = None
            for i in range(0,len(results)):
                if True in results:
                    print("Known")
                    match = known_names[results.index(True)]
                    print(f' - {match} from {results}')
                    color = name_to_color(match)
                else:
                    print("UnKnown")
                    match = "unknown"
                    print(f' - {match} from {results}')
                    color = name_to_color(match)

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            


    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)