from yolo import YOLO
#from utils.image2coord import to_coord
from utils.realSense import realSenseStream
from time import sleep
from PIL import Image, ImageDraw

detect = YOLO().raw_detect_image
take_image = realSenseStream('img').take_image

while True:
	sleep(5)
	path, _ = take_image()
	image = Image.open(path)
	draw = ImageDraw.Draw(image)
	data = detect(image)
	for box in data['objects']:
		# to_coord(box['box']['x'], box['box']['y'])
		print(box['box']['x'], box['box']['y'], box['score'])
		draw.rectangle([box['box']['left'], box['box']['top'], box['box']['right'], box['box']['bottom']], outline=(256,0,0))
	del draw
	image.save('./result/' + path.split('/')[-1])