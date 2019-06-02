#import numpy as np
import math
from math import cos, sin
from serial import Serial
import random
import struct

#ser = Serial('/dev/ttyUSB0', 9600, timeout=100)

physic = {'x': 200, 'y': 200}
# pulse = {'x': 160, 'y': 160}

def serialIn():
	s = str('')
	while ser.in_waiting:
		s = s + ser.read().decode('ASCII')
		# print (ser.read(1)) # each byte
	# return True
	return s == '000'

def serialOut(x, y):
	# x, y = random.randrange(0,180, 1), random.randrange(0,180, 1)
	x = physic['x']//2 - int(x)
	y = physic['y'] - int(y)
	print ('Machine POV:', y, x)
	ser.write(struct.pack('>B', y))
	ser.write(b' ')
	ser.write(struct.pack('>B', x))
	ser.write(b'\n')

if __name__ == '__main__':
	serialOut()