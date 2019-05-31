import pyrealsense2 as rs
import numpy as np
import cv2
import time

class realSenseStream():
	
	def __init__(self):
		super().__init__()
		self.loop = True
		self.config = rs.config()
		self.pipeline = rs.pipeline()
		self.pv_width = 1920
		self.pv_height = 1080
		self.size_config(self.pv_width, self.pv_height)
        
	def size_config(self, w, h):
		self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)

	def save_image(self, name):
		while True:
			frames = self.pipeline.wait_for_frames()
			color_frame = frames.get_color_frame()

			if not color_frame:
				continue
			# Convert images to numpy arrays	
			color_image = np.asanyarray(color_frame.get_data())
			cv2.imwrite('./' + name + '.jpeg', color_image)
			cv2.waitKey(30)
			break

	def open_camera_stream(self):
		self.pipeline.start(self.config)

	def close_camera_stream(self):
		self.pipeline.stop()

rss = realSenseStream()
rss.open_camera_stream()
for i in range(10):
	rss.save_image(str(i))
	time.sleep(1)
rss.close_camera_stream()
