import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np

import face_recognition

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

image = face_recognition.load_image_file(r"C:\Users\LENOVO\Desktop\Projects\webcam\Headset data\No headsets\No headsets6.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

known_encodings=[face_encoding]
known_names = ["Authorized User"]

face_locations = []
face_encodings = []
face_names =[]
process_this_frame = True

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        break

    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    rgb_small_frame = small_frame[:,:,::-1]
    
    # apply object detection
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

    print(bbox, label, conf)

    # draw bounding box over detected objects
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)


        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unauthorized User"


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

    else:
        cv2.putText(frame, 'NO person', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    

    # display output
    cv2.imshow("Real-time object detection", out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows() 
