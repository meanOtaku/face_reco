import face_recognition
import cv2
import pickle
KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  
video = cv2.VideoCapture(0)
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
with open('dataset_faces.dat','rb') as f:
    known_faces = pickle.load(f)
with open('dataset_names.dat','rb') as f:
    known_names = pickle.load(f)
print('Processing unknown faces...')
while True:
    ret, image = video.read()
    locations = face_recognition.face_locations(image,  model='cnn')
    encodings = face_recognition.face_encodings(image, locations)
    for face_encoding, face_location in zip(encodings, locations):
        for i in range(0,len(known_faces)):  
            results = face_recognition.compare_faces(known_faces[i], face_encoding, TOLERANCE)
            match = None
            color = 0
            if True in results:  
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')
                color = name_to_color(match)
            else:
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
            print(f"round = {i}")
    cv2.imshow("filename", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
