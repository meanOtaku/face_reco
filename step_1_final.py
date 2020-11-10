import face_recognition
import os
import cv2
import pickle

folder = "known_faces"
vidcap = cv2.VideoCapture(0)
vidcap.set(cv2.CAP_PROP_FPS, 1)
def getFrame(sec, name):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(f"known_faces/{name}/"+str(count)+".jpg", image)     # save frame as JPG file
        cv2.imshow(f"{name}", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            os.remove(f"F:/college/work/workspace/face reco/known_faces/{name}/1.jpg")
            cv2.destroyWindow(f"{name}")
            hasFrames = False
    return hasFrames
name = input("enter the name : ")
os.mkdir(f"F:/college/work/workspace/face reco/known_faces/{name}")
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec, name)
while success:
    
    
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec, name)








KNOWN_FACES_DIR = 'known_faces'
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
        ima = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        image = image_resize(ima, height = 800)
        encoding = face_recognition.face_encodings(image)
        known_faces.append(encoding)
        known_names.append(name)

        


    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(known_faces,f)
    with open('dataset_names.dat', 'wb') as f:
        pickle.dump(known_names,f)