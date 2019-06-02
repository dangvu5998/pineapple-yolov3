from sympy import Point3D
from sympy import Line3D
from sympy import Plane
from math import sin, cos, tan, radians

import pyrealsense2 as rs
import time
import numpy as np
import cv2
from PIL import ImageDraw, Image

def estimateCoordinate(A, alpha, beta, h, W, ximg, yimg):
	B = Point3D(0.0, 0.0, 0.0)
	C = Point3D(20.0, 0.0, 0.0)
	Z = Point3D(10.0, 10.0, 0.0)
	# pABC = CG3dPlanePN(A, B, C)
	scale = (2 * tan (beta/2) * h / sin(alpha)) / W
	# scale = 1
	xL = (ximg*scale*sin(alpha))
	yL = (yimg*scale)
	zL = (ximg*scale*cos(alpha))
	L = Point3D(xL, yL, zL)
	LineAL = Line3D(A, L)
	planeOxy = Plane(B, C, Z)
	result = LineAL.intersection(planeOxy)
	return result





pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)


# Map depth to color
for i in range(2):
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	A = Point3D(100.0, 0.0, 120.0)
	alpha = radians(25.5)
	beta = radians(69.4)
	W = 848
	h = 80
	ximg, yimg = 300, 300
	point = estimateCoordinate(A, alpha, beta, h, W, ximg, yimg)
	print('%0.5f %0.5f %0.5f'%(point[0][0], point[0][1], point[0][2]))
	color_image = np.asanyarray(color_frame.get_data())
	imgName = str(int(time.time()*1000.0)) + '.jpeg'
	cv2.imwrite(imgName, color_image)
	time.sleep(2)
	image = Image.open(imgName)
	draw = ImageDraw.Draw(image)
	# draw.rectangle([ximg-10, yimg-10, ximg+10, yimg+10], outline = (256,0,0))
	# draw.rectangle([xmimg-10, ymimg-10, xmimg+10, ymimg+10], outline = (256,0,0))
	del draw
	image.save(imgName)
	time.sleep(2)

pipeline.stop()