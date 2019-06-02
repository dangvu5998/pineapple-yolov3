import pyrealsense2 as rs
import time
import numpy as np
import cv2
from PIL import ImageDraw, Image

pipeline = rs.pipeline()
config = rs.config()
w, h = 848, 480
config.enable_stream(rs.stream.depth,  w, h, rs.format.z16, 15)
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 15)

pipe_profile = pipeline.start(config)

frames = pipeline.wait_for_frames()
time.sleep(5)

# Map depth to color
for i in range(1):
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	# Intrinsics & Extrinsics
	# depth_stream = rs.video_stream_profile(pipe_profile.get_stream(rs.stream.depth))
	# intrin = depth_stream.get_intrinsics()
	depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
	color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
	depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
	# print('intrin', intrin)
	print('depth_intrin:', depth_intrin)
	print('color_intrin:', color_intrin)
	print('depth_to_color_extrin', depth_to_color_extrin)
	# Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
	depth_sensor = pipe_profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	print('depth_scale', depth_scale)

	depth_pixel = [620, 250]   # Random pixel
	depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
	color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
	color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
	print('depth_point:', depth_point[0]*100000, depth_point[1]*100000)
	# print('color_point:', color_point)
	# print('color_pixel:', color_pixel)
	color_image = np.asanyarray(color_frame.get_data())
	imgName = './res/' + str(int(time.time()*1000.0)) + '.jpeg'
	cv2.imwrite(imgName, color_image)
	time.sleep(2)
	image = Image.open(imgName)
	draw = ImageDraw.Draw(image)
	draw.rectangle([depth_pixel[0]-10, depth_pixel[1]-10, depth_pixel[0]+10, depth_pixel[1]+10], outline = (256,0,0))
	draw.rectangle([w/2-10, h/2-10, w/2+10, h/2+10], outline = (256,0,0))
	del draw
	image.save(imgName)
	time.sleep(2)

pipeline.stop()
