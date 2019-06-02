import pyrealsense2 as rs
import numpy as np
import cv2
import time

class realSenseStream():
    def __init__(self, path = './', w = 1920, h = 1080):
        super().__init__()
        self.config = rs.config() # Assign to realsense configuration
        self.pipeline = rs.pipeline() # Assign to realsense pipeline
        self.path = path
        if self.path[-1] != '/':
            self.path = self.path + '/'
        self.pv_width = w
        self.pv_height = h
        self.size_config(self.pv_width, self.pv_height)
        self.pipeline.start(self.config)
        
    def size_config(self, w, h):
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)

    def take_image(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            imgName = self.path + str(int(time.time()*1000.0)) + '.jpeg'
            cv2.imwrite(imgName, color_image)
            cv2.waitKey(30)
            return imgName, color_image

    def __del__(self):
        self.pipeline.stop()

if __name__ == '__main__':
    rss = realSenseStream('img')
    for i in range(10):
        rss.take_image()
        time.sleep(1)
