import cv2
import numpy as np

cap = cv2.VideoCapture(0)
minWhiteBoardArea = 100000
outContentColor = [255, 255, 255]
videoSize = (1280, 720)
outputSize = (720, 1280, 3)
blurKernelForDetectingBoard = (9, 9)
blurKernelForBoard = (5, 5)
erosionKernel = (3, 3)
erosionIteration = 2
minHandArea = 2000
textColor = (255, 0, 0)
maxNoiseArea = 20

##########################################
#### creating controls ##################
#########################################
def nothing():
    pass

#cv2.namedWindow("controls")
#cv2.createTrackbar('blur for detecting board', 'controls',0,10,nothing)

##############################################


requiredContours = np.vectorize(lambda x: maxNoiseArea < cv2.contourArea(x) < minHandArea)

# setting video resolutions
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mask = np.zeros(videoSize, np.uint8)  # as the image is grayscale here
while cap.isOpened():
    # read each frame
    ret, colorFrame = cap.read()

    # resize frame
    colorFrame = cv2.resize(colorFrame, videoSize)

    # flip frame
    colorFrame = cv2.flip(colorFrame, -1)

    # create a gray image for the frame
    gray = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

    # apply blur
    blur = cv2.GaussianBlur(gray, blurKernelForDetectingBoard, 0)

    # apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # find contours
    img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find largest contour which may be the board
    large = max(contours, key=cv2.contourArea)

    if cv2.contourArea(large) > minWhiteBoardArea:

        # fit largest contour to a polygon
        epsilon = 0.1 * cv2.arcLength(large, True)
        approx = cv2.approxPolyDP(large, epsilon, True)

        # get number of verticies of the polygon
        vertices_count = len(approx)
        print(f"Area {cv2.contourArea(large)} number of vertices {vertices_count}", end='\r')

        # if it is a rectangle create mask around it
        if vertices_count == 4:
            mask = np.zeros(colorFrame.shape[:2], np.uint8)  # as the image is grayscale here
            imageAddedMask = cv2.drawContours(mask, [approx], 0, 255, -1)
            x, y, w, h = cv2.boundingRect(approx)

    # select only the rectangle region as ROI
    req = cv2.bitwise_and(gray, gray, mask=imageAddedMask)
    res = req[y:y + h, x:x + w]

    # gray board original
    # cv2.imshow("board", res)

    # apply blur on the board
    res = cv2.GaussianBlur(res, blurKernelForBoard, 0)

    # apply Triangle threshold on the board
    ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

    # find contours on board
    boardImg, boardContours, boardHierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    solid = np.full((thresh.shape[0], thresh.shape[1], 3), outContentColor, dtype=np.uint8)

    # add threshold on solid
    final = cv2.drawContours(solid, boardContours, -1, textColor, -1)

    # visualise the result with hands
    # cv2.imshow("with hands", final)

    # removing hand contours and noise
    boardContours = np.array(boardContours)
    required = requiredContours(boardContours)
    boardContours = boardContours[required]

    # create a solid color image
    solid = np.full(outputSize, outContentColor, dtype=np.uint8)
    X = (outputSize[0] - thresh.shape[0]) // 2
    Y = (outputSize[1] - thresh.shape[1]) // 2
    rroi = solid[X:X + thresh.shape[0], Y:Y + thresh.shape[1]]
    # add threshold on solid
    final = cv2.drawContours(rroi, boardContours, -1, textColor, -1)

    # visualise the result in full output resolution
    cv2.imshow("without hands", solid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
