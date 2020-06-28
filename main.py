import face_recognition
import cv2
import numpy as np

import cvlib as cv
from cvlib.object_detection import draw_bbox

from keras.models import load_model

# load the head phone detection model
model = load_model('keras_model.h5')

# load the liveness dectection model 
live_model = load_model('Liveness_detection.h5')

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Could not open webcam")
    exit()

# Load a sample picture and learn how to recognize it.
image = face_recognition.load_image_file(r"C:\Users\LENOVO\Desktop\Projects\webcam\Headset data\No headsets\No headsets6.jpg")
face_encoding = face_recognition.face_encodings(image)[0]


# Create arrays of known face encodings and their names
known_encodings = [
    face_encoding,
    
]
known_names = [
    "Maheer",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # for head phone detection
    h_frame =cv2.resize(frame,(224,224)) 
    h_frame = h_frame.reshape(1,224,224,3) 

    # for liveness detection
    l_frame = cv2.resize(frame,(32,32))
    l_frame=l_frame.reshape(1,32,32,3)

    # for Face recognition
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        predicted = model.predict_classes(h_frame)

        # object detection model used yolov3 model due to low RAM config, we can use other models which gives beter output
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

        print(bbox, label, conf)

        if (label ==["cell phone"]):
            cv2.putText(frame, 'Object Phone detected', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        """elif (label ==["bottle"]):
            cv2.putText(frame, 'Object Bottle Detected', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        elif (label ==["pen"]):
            cv2.putText(frame, 'Object Pen Detected', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        elif (label ==["book"]):
            cv2.putText(frame, 'Object Book Detected', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)"""
            
        # draw bounding box over detected objects
        out = draw_bbox(frame, bbox, label, conf, write_conf=True)

        
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unauthorized User"

            # To predict Liveness 
            live_prediction=live_model.predict_classes(l_frame)


            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    if face_locations:       
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            

            if len(face_locations)>1:
                cv2.putText(frame, 'Another person Detected', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                

        
            
            if(live_prediction==0):
                image = cv2.putText(frame,"No Liveness",(00,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            else:
                image = cv2.putText(frame,"Liveness",(25,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            
                
            if:
                image = cv2.putText(frame,"HeadPhones detected",(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            else:
                image = cv2.putText(frame,"no headphones detected",(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    else:
        cv2.putText(frame, 'NO person', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(frame, 'NO Liveness',(25,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # Display the resulting image
    cv2.imshow('Video', out)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
