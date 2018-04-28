import cv2
import requests
import numpy as np


def rectangle_detection(image, gray):  
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    (thresh, im_bw) = cv2.threshold(im_bw, 1, 2, 0)
    cv2.imwrite("bw", im_bw)

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    cv2.imwrite("cnt_", image)

r = requests.get('http://192.168.1.198/mjpg/video.mjpg', auth=('root', 'root'), stream=True)
print("Opening the stream...")
if(r.status_code == 200):
    bytes = bytes()
    for chunk in r.iter_content(chunk_size=1024):
        bytes += chunk
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.medianBlur(i,5)
            imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cimg = cv2.cvtColor(imgg,cv2.COLOR_GRAY2BGR)
            circles = cv2.HoughCircles(imgg,cv2.HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=1,maxRadius=30)
            if circles is None:
                    continue
            print (circles)
            #circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

            cv2.imshow("preview", cimg)
            
            if cv2.waitKey(1) == 27:
                cv2.destroyWindow("preview")
                exit(0)
else:
    print("Received unexpected status code {}".format(r.status_code))