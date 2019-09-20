import cv2
import numpy as np
import os
anno_file = "wider_face_train.txt"
im_dir = "WIDER_train/images"

with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
mean_buf = np.array([0.0,0.0,0.0])
index = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    #bbox = map(float, annotation[1:]) 
    #boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    mean_buf += img.mean(0).mean(0)
    index += 1
    print("%d:%d\r\n" % (index,num))

mean_ret = mean_buf  / num
print(mean_ret)
