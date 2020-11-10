import face_recognition
import os
import cv2


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.47
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []
#image = []


for name in os.listdir(KNOWN_FACES_DIR):
    print("name = ", name)

    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print("filename = ", filename)

        ima = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        #ima = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE) 

        #r = 100.0 / ima.shape[1]
        #dim = (100, int(ima.shape[0] * r))

        #image = cv2.resize(ima, dim, interpolation = cv2.INTER_AREA) 
        #ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        image = image_resize(ima, height = 800)
        encoding = face_recognition.face_encodings(image)
        
        known_faces.append(encoding)
        known_names.append(name)


print('Processing unknown faces...')
for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    print(f'Filename {filename}', end='')
    ima = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    #ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    #image = cv2.resize(ima, (800,600)) 
    image = image_resize(ima, height = 800)

    locations = face_recognition.face_locations(image,  model='cnn')

    encodings = face_recognition.face_encodings(image, locations)


    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        for i in range(0,len(known_faces)):  #loops through each encoded known_face 

            results = face_recognition.compare_faces(known_faces[i], face_encoding, TOLERANCE)


            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = name_to_color(match)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)