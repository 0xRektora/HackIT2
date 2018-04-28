import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import firebase_admin
from firebase_admin import credentials
import pyrebase
import json, codecs
import threading




lock = False
last_modified = time.time()
img_rgb = 0
data = {}
run = True
cred = config = {
  "apiKey": "AIzaSyClaMtVMWFEZyOo2ArfSvlRd0QB77RJRns",
  "authDomain": "hacj-cd17a.firebaseapp.com",
  "databaseURL": "https://hacj-cd17a.firebaseio.com",
  "storageBucket": "hacj-cd17a.appspot.com",
  "serviceAccount": "firebase_sdk.json"
}
firebase = pyrebase.initialize_app(cred)
db = firebase.database()
#data = {"P1":1,"P2":1}
#db.child("parking").update(data)



matrice = np.zeros((2, 4))
points = (0,0)
#Matrix
def refresh_matrice(shape):
    a = np.zeros(shape)
    return a

def jsonify():
    myjson = {"P1": int(matrice.item(0,0)), "P2": int(matrice.item(0,1)), "P3": int(matrice.item(0,2)), "P4": int(matrice.item(0,3)), "P5": int(matrice.item(1,0)), \
                        "P6": int(matrice.item(1,1)), "P7": int(matrice.item(1,2)), "P8": int(matrice.item(1,3))}
    return myjson

def send_firebase():
    data = jsonify()
    db.child("parking").update(data)
    print(matrice)
    
    
def update_last_modified():
    last_modified = time.time()

#define points saw on the matrix
def get_point(frame_w, frame_h ,x , y, offset):
    row = 0
    column = 0
    if x < (frame_w/4)+offset:
        column = 0
    elif x < (frame_w/3)+offset and (x > frame_w/4)+offset:
        column = 1
    elif x > (frame_w/2)+offset and x < (frame_w/2)+(frame_w/4)+offset:
        column = 2
    elif x < frame_w +offset and x > (frame_w/2)+(frame_w/4)+offset:
        column = 3
    
    if y < (frame_h/2 )+offset:
        row = 0
    elif y > (frame_h/2)+offset:
        row = 1
    
    
    return (row,column)


if __name__ == "__main__":
    img = cv2.VideoCapture("http://root:root@192.168.1.198/mjpg/video.mjpg")#http://root:root@192.168.1.198/mjpg/video.mjpg
    #worker = threading.Thread(target=send_firebase, daemon=True)
    
    #img = cv2.imread("Parking4.jpg")
    while True:    
        matrice = refresh_matrice((2,4))
        rval, frame = img.read()
        img_rgb = frame
        #img_rgb = img
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('final.png',0)
        w, h = template.shape[::-1]
        imgW, imgH = (img_rgb.shape[1:2][0], img_rgb.shape[0:1][0]) 

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        (minValue, maxValue, minLocation, maxLocation) = cv2.minMaxLoc(res)
        threshold = 0.8
        
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
            try:
                points = get_point(imgW, imgH, pt[0], pt[1], 0)
                #print("points : ", points)
                matrice.itemset(points, 1)
            except Exception as e:
                print(e)
        
        #print(matrice)
        #send_firebase()
        #print("X & Y of the webcam : ", img.shape[:2], "X & Y of the match", minLocation)
        send_firebase()
        #cv2.imshow('res.png',img_rgb)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            run = False
            #worker.join()
            break
        
        