import cv2
import os
from utils.pineapple_segment import ripe_pineapple_segment

img_dir_path = '/media/trivu/data/DataScience/ComputerVision/dua/new_data/cropped_pineapple'
img_fns = os.listdir(img_dir_path)
img_fns = [img_fn for img_fn in img_fns if img_fn.startswith('full')]
i = 0
window_original_name = 'Original Image'
window_threshold_name = 'Threshold Image'
window_img_masked = 'Masked image'

last_i = -1
while True:
    key = cv2.waitKey(20)
    if key == 27:
        cv2.destroyAllWindows()
        break
    elif key == ord('n'):
        i += 1
    elif key == ord('p'):
        i -= 1
    if i == last_i:
        continue
    else:
        last_i = i
    i = i%len(img_fns)
    img = cv2.imread(os.path.join(img_dir_path, img_fns[i]))
    img_th = ripe_pineapple_segment(img, img_type='BGR')
    img_masked = cv2.bitwise_and(img, img, mask=img_th)
    cv2.imshow(window_original_name, img)
    cv2.imshow(window_threshold_name, img_th)
    cv2.imshow(window_img_masked, img_masked)


