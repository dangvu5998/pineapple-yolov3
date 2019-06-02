from yolo import YOLO
from utils.serialIO import serialIn, serialOut
from utils.image2coord import to_coord
from utils.realSense import realSenseStream
from time import sleep
from PIL import Image, ImageDraw

detect = YOLO().raw_detect_image
take_image = realSenseStream('img').take_image

while True:
	while serialIn() == False:
		sleep(2)
	path, _ = take_image()
	image = Image.open(path)
	draw = ImageDraw.Draw(image)
	data = detect(image)
	for box in data['objects']:
		if box['class'] == 2:
			print('Box x/y/score:', int(box['box']['x']), int(box['box']['y']), box['score'])
			rx, ry = to_coord(box['box']['x'], box['box']['y'])
			serialOut(rx, ry)
			draw.rectangle([box['box']['left'], box['box']['top'], box['box']['right'], box['box']['bottom']], outline=(255,0,0), width=10)
		# elif box['class'] == 3:
		# 	draw.rectangle([box['box']['left'], box['box']['top'], box['box']['right'], box['box']['bottom']], outline=(0,0,255), width=10)
		while serialIn() == False:
			sleep(2)
	del draw
	image.save('./result/' + path.split('/')[-1])
