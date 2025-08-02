import cv2
import threading
import time
# import Find_Center

class Camera:
    def __init__(self, src=0, shape=[640, 480]):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, shape[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[1])
        self.stopped = False
        for _ in range(10):  # warm up the camera
            (self.grabbed, self.frame) = self.stream.read()

    def start(self):
        threading.Thread(target=self.update, args=()).start()

    def start(self):
        self.stopped = False
        threading.Thread(target=self.update, args=()).start()
        time.sleep(0.5)

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        # self.stream.release()


if __name__ == "__main__":
    camera = Camera(0)
    camera.start()
    while True:
        img = camera.read()
        cv2.imshow("img", img)
        cv2.waitKey(1)
