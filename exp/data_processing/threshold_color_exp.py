from __future__ import print_function
import cv2
import os
import json

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Original Image'
window_detection_name = 'Detected Image'
window_img_masked = 'Masked image'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

## [low]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
## [low]

## [high]
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
## [high]

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

## [window]
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
## [window]

## [trackbar]
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
## [trackbar]

img_dir_path = '/media/trivu/data/DataScience/ComputerVision/dua/new_data/cropped_pineapple'
img_fns = os.listdir(img_dir_path)
i = 0
HSV_threshold_f = open('HSV_threshold.txt', 'a')
while True:
    ## [while]
    i = i%len(img_fns)
    img = cv2.imread(os.path.join(img_dir_path, img_fns[i]))
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_th = cv2.inRange(img_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    img_masked = cv2.bitwise_and(img, img, mask=img_th)
    ## [while]

    ## [show]
    cv2.imshow(window_capture_name, img)
    cv2.imshow(window_detection_name, img_th)
    cv2.imshow(window_img_masked, img_masked)
    ## [show]

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break
    if key == ord('s'):
        HSV_threshold_f.write(json.dumps({
            'low_H':low_H,
            'high_H': high_H,
            'low_S': low_S,
            'high_S': high_S,
            'low_V': low_V,
            'high_V': high_V
            }) + '\n')
    if key == ord('n'):
        i += 1
    if key == ord('p'):
        i -= 1
HSV_threshold_f.close()
