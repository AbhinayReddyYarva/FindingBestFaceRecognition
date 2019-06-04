# import the necessary packages
from imutils import face_utils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
#p = "shape_predictor_5_face_landmarks.dat"
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
#cap.set(5, 20)
#framerate = cap.get(5)
#print(framerate)

#_, image = cap.read()
#videoname = 'video281.avi'
#videotype = cv2.VideoWriter_fourcc(*'XVID') #fourcc
#out = cv2.VideoWriter(videoname, videotype, 10, (640, 480))
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        print(i) # Working on Number of faces in each frame
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        count = 0
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            count += 1
            if count == 1:
                x1 = x
            elif count == 9:
                y2 = y
            elif count == 17:
                x2 = x
            elif count == 28:
                y1 = y
        cv2.rectangle(image, (x1 - 15, y1 - 80), (x2 + 15, y2 + 20), (0, 255, 0), 2)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    #out.write(image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
