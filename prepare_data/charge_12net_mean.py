import sys
sys.path.append('../demo/')
sys.path.append('.')
sys.path.append('/home/bitmain/lian.he/caffe/caffe/python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np
import os
import random
from utils import *
#deploy = '../12net/12net.prototxt'
#caffemodel = '../12net/12net.caffemodel'
deploy = '../12net/12net.prototxt'
caffemodel = '../12net/models_mean/solver_iter_500000.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()
def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = img.copy()-128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        caffe_img_2 = img.copy()
        scale_img_2 = cv2.resize(caffe_img_2,(ws,hs))
        scale_img_2 = np.swapaxes(scale_img_2, 0, 2)
        scale_img_2 = scale_img_2.astype(np.float32)
        scale_img_2[0,:,:]  = (scale_img_2[0,:,:] - 104.146) / 127.5
        scale_img_2[1,:,:]  = (scale_img_2[1,:,:] - 110.807) / 127.5
        scale_img_2[2,:,:]  = (scale_img_2[2,:,:] - 119.856) / 127.5
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img_2
        caffe.set_device(0)
        caffe.set_mode_gpu()
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    return rectangles
anno_file = 'wider_face_train.txt'
im_dir = "WIDER_train/images/"
#anno_file = 'wider_face_tmp.txt'
#im_dir = "WIDER_val/images/"
#save_dir = "../24net/24"
#neg_save_dir  = "../24net/24/negative"
#pos_save_dir  = "../24net/24/positive"
#part_save_dir = "../24net/24/part"
#
#ensure_directory_exists(save_dir)
#ensure_directory_exists(neg_save_dir)
#ensure_directory_exists(pos_save_dir)
#ensure_directory_exists(part_save_dir)
#
image_size = 24
#f1 = open('../24net/24/pos_24.txt', 'w')
#f2 = open('../24net/24/neg_24.txt', 'w')
#f3 = open('../24net/24/part_24.txt', 'w')
threshold = [0.7,0.6,0.7]
with open(anno_file, 'r') as f:
    annotations = f.readlines()
annotations=annotations[:20]
num = len(annotations)
print("%d pics in total" % num)

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
image_idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    bbox = map(float, annotation[1:])
#    print(bbox)
#    print(annotation)
#    gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    gts = np.array(annotation[1:], dtype=np.float32).reshape(-1, 4)

    img_path = im_dir + annotation[0] + '.jpg'
    rectangles = detectFace(img_path,threshold)
    img = cv2.imread(img_path)
    image_idx += 1
    #view_bar(image_idx,num)
    bad_count = 0
    part_count = 0
    pos_count = 0
    for box in rectangles:
        x_left, y_top, x_right, y_bottom, _ = [int(i) for i in box]
        crop_w = x_right - x_left + 1
        crop_h = y_bottom - y_top + 1
        # ignore box that is too small or beyond image border
        if crop_w < image_size or crop_h < image_size :
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gts)
        # save negative images and write label
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            bad_count += 1
            n_idx += 1
        else:
            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.65:
                pos_count += 1
                p_idx += 1

            elif np.max(Iou) >= 0.4:
                part_count += 1
                d_idx += 1
    print(pos_count,part_count,bad_count)


print("total:",p_idx,d_idx,n_idx)
#f1.close()
#f2.close()
#f3.close()
