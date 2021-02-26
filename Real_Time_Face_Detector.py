# from cv2 import cv2 
import cv2 
from random import randrange
# import keyboard


#Load some per-trained data on a face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Stream a video to detect faces in
webcam = cv2.VideoCapture(0)#(0) means access a default webcam / (video_example.mp4) means accessing the specified video 

# Iterate forever over the frames
while True:
       ### Read the current frame
       successful_frame_read, frame = webcam.read() #successful_frame_read [it means that if the frame was read successfully, then read the frame <but this is of no use at all because it will always be true.

       # must convert to grayscale
       grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          
       #Detect faces
       face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
       
       #Draw rect around the faces
       for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x, y),(x+w, y+h),(0,randrange(128,256),0),5)

       
       cv2.imshow("Clever Programmer Face Dectector", frame)
       cv2.waitKey(1)#This means that the code automatically waits 1 millisecond #In openCV, you can't display anything without a wait key
       # Must convert image to grayscale : (this helps computer to recognise image easily with only few colors):
       
       #### Stop if Q key is pressed              
       # if Key==81 or Key==113:
       # break
       # webcam.release()
print("Code Complete!") 

# 1:5:00      