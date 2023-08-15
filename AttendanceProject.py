# importing the packages
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
Images = []
classNames = []
myList = os.listdir(path)
print (myList)
# to add the name without the file format
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg) #inserting one by one in to the Imagelist
    classNames.append(os.path.splitext(cl)[0])
print (classNames)

#function for finding the encodings for each image
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode) #inserting one by one in to the encodelist
    return encodeList

# marking Attendance
def markAttendance(name):
    with open ('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # if name already entered it shouldnt re-enter it
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(Images)
print('Encoding Complete')

# taking the test image from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None, 0.25, 0.25) #resizing the img
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #converting to RGB
    facesCurFrame = face_recognition.face_locations(imgS) #finding the face locations for one or multiple images present in the web cam
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #finding the face encodings for one or multiple images present in the web cam


    for encodeFace, faceLoc in zip (encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) #comparing the faces one by one
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4,x2 * 4,y2 * 4,x1 * 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
