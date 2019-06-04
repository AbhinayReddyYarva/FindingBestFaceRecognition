# import the necessary packages
import dlib
import cv2
import time
 
# initialize dlib's face detector (MMOD-based)
p = "mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(p)

inWidth = 0
inHeight = 300

cap = cv2.VideoCapture(0)
cap.set(5, 30)
frameRate = cap.get(5)
print(frameRate)

while True:
    frameId = cap.get(1) #current frame number--this is for frame rate
    print(frameId)
    # load the input image and convert it to grayscale
    _, image = cap.read()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #if (frameId % (math.floor(frameRate/30)) == 0):
    t = time.time()
    print(t)
    frameDlibMMOD = image.copy()
    frameHeight = frameDlibMMOD.shape[0]
    frameWidth = frameDlibMMOD.shape[1]
    inWidth = int((frameWidth / frameHeight)*inHeight)
    
    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth
    
    frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))
    
    frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
    
    # detect faces in the image
    rects = detector(frameDlibMMODSmall, 0)
    
    print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        print(i)
        # determine the face region, then
        # take the roi-coordinates to a NumPy
        # array
        cvRect = [int(rect.rect.left()*scaleWidth), int(rect.rect.top()*scaleHeight),
                  int(rect.rect.right()*scaleWidth), int(rect.rect.bottom()*scaleHeight) ]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/150)), 4)
        timetaken = time.time() - t
        print(time.time())
        print(timetaken)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("frameDlibMMOD",frameDlibMMOD)
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
