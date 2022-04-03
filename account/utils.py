import os
from xmlrpc.client import boolean
import cv2
import json
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


cap = cv2.VideoCapture(0)
dictionary = {}



def detect_face():
    # load model
    model = model_from_json(open("static/ml/gabor.json", "r").read())
    #load weights
    model.load_weights('static/ml/gabor.h5')

    face_haar_cascade = cv2.CascadeClassifier('static/ml/haarcascade_frontalface_default.xml')


    


    

    frame = 0
    a = True
    while a:
        # captures frame and returns boolean value and captured image
        ret, test_img = cap.read()

        if not ret:
            break

        #rotate video
        #test_img = cv2.rotate(test_img, cv2.cv2.ROTATE_90_CLOCKWISE)
        #convert to grayscale
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
            # cropping region of interest i.e. face area from  image
            roi_gray = gray_img[y:y+w, x:x+h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions1 = ('angry', 'disgust', 'fear', 'happy','sad', 'surprise', 'neutral')
            predicted_emotion = emotions1[max_index]

            #Save every nth frame result
            n = 10
            if frame%n:
                dictionary[frame] = predicted_emotion

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        frame += 1

        #if cv2.waitKey(10) == ord('q'):
               # wait until 'q' key is pressed
            #break

    
    

    


def close_cam1():

    cap.release()
    cv2.destroyAllWindows

    emotions = ['angry', 'disgust', 'fear', 'happy','sad', 'surprise', 'neutral']
    #Save to json
    with open("static/json/sample.json", "w") as outfile:json.dump(dictionary, outfile)


    #Find emotion with highest frequency in video
    from collections import Counter

    listA  = dictionary.values()

    occurence_count = Counter(listA)
    results = dict(occurence_count)

    listb = list(results.keys())

    print(listb)



    for i in emotions:
        if i not in listb:
            results[i] = 1
        
    print(results)
    
    res=occurence_count.most_common(1)[0][0]
    #print("Element with highest frequency:\n",res)

    return results