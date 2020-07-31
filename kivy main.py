from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np

class KivyCamera(Image):
    def __init__(self,capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0/fps)
        self.minWhiteBoardArea = 100000
        self.outContentColor = [255, 255, 255]
        self.videoSize = (1280, 720)
        self.outputSize = (720, 1280, 3)
        self.blurKernelForDetectingBoard = (9, 9)
        self.blurKernelForBoard = (5, 5)
        self.erosionKernel = (3, 3)
        self.erosionIteration = 2
        self.minHandArea = 2000
        self.textColor = (255, 0, 0)
        self.maxNoiseArea = 20
        self.mask = np.ones(self.videoSize, np.uint8)  # as the image is grayscale here

        #rectangle requirements
        self.X, self.Y, self.W, self.H = 0,0,self.videoSize[0], self.videoSize[1]
    def update(self, dt):
        requiredContours = np.vectorize(lambda x: self.maxNoiseArea < cv2.contourArea(x) < self.minHandArea)

        ret, colorFrame = self.capture.read()
        buf = colorFrame.tostring()
        image_texture = Texture.create(
            size=(colorFrame.shape[1], colorFrame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture
        if ret:
            #resize
            colorFrame = cv2.resize(colorFrame, self.videoSize)

            #flip
            colorFrame = cv2.flip(colorFrame, 1)

            #create gray version of the frame
            gray = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

            #apply blur
            blur = cv2.GaussianBlur(gray, self.blurKernelForDetectingBoard, 0)

            #find edges
            edges = cv2.Canny(gray, 100, 200)

            #find contours
            img, contours, _ = cv2.findContours(edges,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            #find largest contour which may be the board
            large = max(contours, key=cv2.contourArea)

            if cv2.contourArea(large) > self.minWhiteBoardArea:

                #fit largest contour to a polygon
                epsilon = 0.1 * cv2.arcLength(large, True)
                approx = cv2.approxPolyDP(large, epsilon, True)

                #get vertices count of the polygon
                vertices_count = len(approx)


                if vertices_count == 4:
                    self.mask = np.zeros(gray.shape, np.uint8)  # as the image is grayscale here
                    cv2.drawContours(self.mask, [approx], 0, 255, -1)
                    self.X,self.Y,self.W,self.H = cv2.boundingRect(approx)

            #select only the boardContour rectangle
            req = cv2.bitwise_and(gray, gray, mask=self.mask)
            res = req[self.Y:self.Y + self.H, self.X:self.X + self.W]

            # apply blur on the board
            res = cv2.GaussianBlur(res, self.blurKernelForBoard, 0)

            # apply Triangle threshold on the board
            ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

            # find contours on board
            boardImg, boardContours, boardHierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # removing hand contours and noise
            boardContours = np.array(boardContours)
            required = requiredContours(boardContours)
            boardContours = boardContours[required]

            # create a solid color image
            solid = np.full(self.outputSize, self.outContentColor, dtype=np.uint8)
            X = (self.outputSize[0] - thresh.shape[0]) // 2
            Y = (self.outputSize[1] - thresh.shape[1]) // 2
            rroi = solid[X:X + thresh.shape[0], Y:Y + thresh.shape[1]]
            # add threshold on solid
            final = cv2.drawContours(rroi, boardContours, -1, self.textColor, -1)
            #convert it to texture
            buf =solid.tostring()
            image_texture = Texture.create(
                size=(solid.shape[1], solid.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture



class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    CamApp().run()