from flask import Flask, request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

path = r'F:\xampp\htdocs\video\Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')

encodeListKnown = findEncodings(images)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        path = r'F:\xampp\htdocs\video\Images_Attendance'
        images = []
        classNames = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

        def markAttendance(name):
            with open('Attendance.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    time_now = datetime.now()
                    tString = time_now.strftime('%H:%M:%S')
                    dString = time_now.strftime('%d/%m/%Y')
                    f.writelines(f'\n{name},{tString},{dString}')

        encodeListKnown = findEncodings(images)

        video = request.files['video']
        print("video recieved")
        video_file = video.read()
        np_video = np.frombuffer(video_file, np.uint8)
        with open('temp.mp4', 'wb') as f:
            f.write(np_video)

        # Initialize the video capture process
        video = cv2.VideoCapture('temp.mp4')
        #video = cv2.VideoCapture(np_video)
        attendance = []
        print("flag 0 passed ")
        while True:
            success, img = video.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)



            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(img)
            encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)
            print("flag 1 passed")
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                matchIndex = np.argmin(faceDis)
                print("flag 2 passed")
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print("flag 3 passed")
                    video.release()
        #return jsonify({"attendance": attendance})
            return name
    except Exception as e:
        return("An error occurred:", str(e))

if __name__ == '__main__':
    app.run(debug=True)