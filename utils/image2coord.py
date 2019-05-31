import numpy as np
import math
from math import cos, sin
from serial import Serial

ser = Serial("COM1", 9600)
ser.begin()

def to_coord(x, y):	
	fx = 0.00193
	fy = 0.00193
	(a, b, g) = (0, 0, 0)

	ca = cos(a)
	sa = sin(a)

	cb = cos(b)
	sb = sin(b)

	cg = cos(g)
	sg = sin(g)

	pixelMat = np.matrix([[x], [y], [1]])
	transfromedMat = np.matrix([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
	orientedMat = np.matrix(
	  [[1 + cb + cg, -sg, sb],
	   [sg, ca + 1 + cg, -sa],
	   [-sb, sa, ca + cb + 1]])
	tMat = np.linalg.inv(transfromedMat*orientedMat)*pixelMat

	ser.print(tMat[0]/tMat[2])
	ser.print(',')
	ser.printline(tMat[1]/tMat[2])
