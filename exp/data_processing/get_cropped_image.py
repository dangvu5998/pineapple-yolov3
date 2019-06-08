import cv2
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

data_directory_path = '/media/trivu/data/DataScience/ComputerVision/dua/new_data/train'
result_path = '/media/trivu/data/DataScience/ComputerVision/dua/new_data/cropped_pineapple'
labels_path = os.listdir(data_directory_path)
labels_path = [label_path for label_path in labels_path if label_path.endswith('.xml')]
_id = 0
for label_path in tqdm(labels_path):
    tree = ET.parse(os.path.join(data_directory_path, label_path))
    root = tree.getroot()
    filename = root.find('filename').text
    img_path = os.path.join(data_directory_path, filename)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
#        if not obj_name.startswith('full'):
        if not obj_name in ['body ripe pineapple', 'full ripe pineapple']:
            continue
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        xmax = int(bndbox.find('xmax').text)
        img_cropped = img[ymin:ymax, xmin:xmax]
        _id += 1 
        cv2.imwrite(os.path.join(result_path, obj_name+str(_id)+'.jpg'), img_cropped)
#    break 
