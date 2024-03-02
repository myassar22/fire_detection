import cv2
from playsound import playsound

fire_myys = cv2.CascadeClassifier('fire_detection.xml') #  file name
myy = cv2.VideoCapture(0)
while (True):
    ret,frame = myy.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    fire = fire_myys.detectMultiScale(frame,1.2,5)                         # 1.2 fire recognition accuracy 
    # 5 the properties of fire to identify it
    for (x,y,w,h) in fire:
        roi_gray = gray[y:y+h ,  x:x+w]
        roi_color = frame[y:y+h ,  x:x+w]
        print('Fire is detected')
        playsound('audio.mp3')

    cv2.imshow('MYASSAR', frame)
    if cv2.waitKey(1) & 0xFF == ord('m') :  # the latter m to break the programing  
     break
    
