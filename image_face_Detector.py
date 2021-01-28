from cv2 import cv2
from random import randrange


#Load some per-trained data on a face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# choose an image to detect faces from
img = cv2.imread("img/josue.jfif") 
# Instead of capturing image, let's use video
# webcam = cv2.VideoCapture("video.mp4")#(0) means access a default webcam / (video_example.mp4) means accessing the specified video 

# Iterate forever over the frames
# while True:
       ### Read the current frame
    #    successful_frame_read, frame = webcam.read() #successful_frame_read [it means that if the frame was read successfully, then read the frame <but this is of no use at all because it will always be true.

    #    # must convert to grayscale
    #    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    cv2.imshow("Clever Programmer Face Dectector", grayscaled_frame)
    #    cv2.waitKey()
# Must convert image to grayscale : (this helps computer to recognise image easily with only few colors):
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# Draw rectangles around the faces
for(x,y,w,h) in face_coordinates:
   cv2.rectangle(img,(x, y),(x+w, y+h),(0,randrange(128,256),0),5)
#print the coordinates
# print(face_coordinates)


#Display image with the faces
cv2.imshow("Clever Programmer Face Detector", img)
#Waits until you press a key
cv2.waitKey()
print("Code Complete!") 

# 1:5:00      