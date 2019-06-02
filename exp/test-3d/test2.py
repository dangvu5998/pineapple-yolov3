import pyrealsense2 as rs
import time
import numpy as np
import cv2
from PIL import ImageDraw, Image

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)


# Map depth to color
for i in range(3):
	pc = rs.pointcloud()
	frames = pipeline.wait_for_frames()
	depth = frames.get_depth_frame()
	color = frames.get_color_frame()
	img_color = np.asanyarray(color.get_data())
	img_depth = np.asanyarray(depth.get_data())
	pc.map_to(color)
	points = pc.calculate(depth)
	vtx = np.asanyarray(points.get_vertices())
	tex = np.asanyarray(points.get_texture_coordinates())

	npy_vtx = np.zeros((len(vtx), 3), float)
	for i in range(len(vtx)):
	    npy_vtx[i][0] = np.float(vtx[i][0])
	    npy_vtx[i][1] = np.float(vtx[i][1])
	    npy_vtx[i][2] = np.float(vtx[i][2])

	npy_tex = np.zeros((len(tex), 3), float)
	for i in range(len(tex)):
	    npy_tex[i][0] = np.float(tex[i][0])
	    npy_tex[i][1] = np.float(tex[i][1])pc = rs.pointcloud()
	pixel = [300, 300]
	color_image = np.asanyarray(color_frame.get_data())
	imgName = str(int(time.time()*1000.0)) + '.jpeg'
	cv2.imwrite(imgName, color_image)
	time.sleep(2)
	image = Image.open(imgName)
	draw = ImageDraw.Draw(image)
	draw.rectangle([pixel[0]-10, pixel[1]-10, pixel[0]+10, pixel[1]+10], outline = (256,0,0))
	del draw
	image.save(imgName)
	time.sleep(2)

pipeline.stop()