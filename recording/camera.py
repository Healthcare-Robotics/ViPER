import cv2
import time
import os
import numpy as np

class Camera:
    def __init__(self, resolution=480, resource=0, view=False):
        self.vc = cv2.VideoCapture(resource)

        if resolution == 480:
            # setting to 480p (~30 fps)
            self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        elif resolution == 720:
            # setting to 720p (~10 fps)
            self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        elif resolution == 1080:
            # setting to 1080p (~4 fps)
            self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        else:
            print('Invalid camera resolution')
            exit()
  
        self.vc.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.first_frame_time = 0
        self.current_frame_time = 0
        self.frame_count = 0
        self.view = view

    def get_frame(self):
        self.ret, frame = self.vc.read()
        self.current_frame_time = time.time()

        if self.first_frame_time == 0:
            self.first_frame_time = time.time()

        if not self.ret:
            print("frame {} was bad".format(self.frame_count))
            
        self.frame_count += 1

        return frame

if __name__ == "__main__":
    feed = Camera(resolution=480, view=True)
    frame = feed.get_frame()

    while feed.ret:
        frame = feed.get_frame()
        print(frame.shape)
        if feed.view:
            cv2.imshow("frames", frame)
            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break
        print('Average FPS', feed.frame_count / (time.time() - feed.first_frame_time))
        print(feed.frame_count, ' frames captured')