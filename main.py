from yolo import YOLO
from utils.image2coord import to_coord
from utils.realSense import realSenseStream
from time import sleep
from PIL

detect = YOLO(None).raw_detect_image
take_image = realSenseStream().take_image

while True:
    _, image = take_image()
    data = detect(image)
    for box in data['objects']:
        to_coord(box['box']['x'], box['box']['y'])