import sys
sys.path.append('.')
sys.path.append('/home/bitmain/lian.he/caffe/caffe/python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np
from utils import IoU
deploy = '../12net/12net.prototxt'
caffemodel = '../12net/models_mean/solver_iter_500000.caffemodel'
#caffemodel = '12net.caffemodel'
#caffemodel = '../12net/12net-cls-only.caffemodel'

net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '../24net/24net.prototxt'
#deploy = '24net.prototxt'
#caffemodel = '../24net/24net.caffemodel'
caffemodel = '../24net/models_mean/solver_iter_500000.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '../48net/48net.prototxt'
caffemodel = '../48net/models/solver_iter_500000.caffemodel'
#deploy = '48net.prototxt'
#caffemodel = '48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

#
def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
#    caffe_img = (img.copy()-127.5)/127.5
    caffe_img = img.copy()
    caffe_img = caffe_img.astype(np.float32)
    caffe_img[:,:,0]  = (caffe_img[:,:,0] - 104.146) / 127.5
    caffe_img[:,:,1]  = (caffe_img[:,:,1] - 110.807) / 127.5
    caffe_img[:,:,2]  = (caffe_img[:,:,2] - 119.856) / 127.5
    caffe_img2 = (img.copy() -127.5) / 127.5

    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
        caffe.set_device(0)
        caffe.set_mode_gpu()
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
#        print(cls_prob.shape)
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        #print("mean0:%f" % out[i]['prob1'][0][0].mean()) 
        #print("mean0:%f" % out[i]['prob1'][0][1].mean()) 
        #print("shape:",out[i]['prob1'][0][0].shape)
        #print(len(np.where(out[i]['prob1'][0][0] > threshold[0])[0]))
        #print(len(np.where(out[i]['prob1'][0][0] < threshold[0])[0]))
        ##continue
        #print(cls_prob)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    print("-------------")
    print(len(rectangles))
    rectangles = tools.NMS(rectangles,0.7,'iou')
    print(len(rectangles))

    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['fc5-2']
    #roi_prob = out['conv5-2']
    print(len(rectangles))
    print(cls_prob.shape)
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    print(len(rectangles))
#
#    
    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img2[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
#    roi_prob = out['conv6-2']
#    pts_prob = out['conv6-3']
    roi_prob = out['fc6-2']
    pts_prob = out['fc6-3']
    #roi_prob = roi_prob - roi_prob
    #print(roi_prob)
    print(cls_prob.shape)
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
    rectangles = tools.NMS(rectangles,0.7,'iou')
    print(len(rectangles))
#
#####################################################
##  tmp code:
#    if len(rectangles)==0:
#        return rectangles
#    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
#    crop_number = 0
#    bbox = "531.0 215.0 574.0 277.0 363.0 219.0 421.0 296.0 251.0 262.0 320.0 354.0 116.0 252.0 196.0 351.0 707.0 248.0 786.0 338.0 870.0 250.0 960.0 376.0 280.0 124.0 308.0 155.0"
#    boxes = np.array(bbox.split(), dtype=np.float32).reshape(-1, 4)
#    for rectangle in rectangles:
#        print("%d %f" % (crop_number,np.max(IoU(rectangle,boxes))))
#        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
#        cv2.imwrite('./crop_file2/%d.jpg' % crop_number,crop_img*127.5 + 127.5)
#        scale_img = cv2.resize(crop_img,(48,48))
#        cv2.imwrite('./crop_file/%d.jpg' % crop_number,scale_img*127.5 + 127.5)
#        scale_img = np.swapaxes(scale_img, 0, 2)
#        net_48.blobs['data'].data[crop_number] =scale_img 
#        crop_number += 1
#    out = net_48.forward()
#    cls_prob = out['prob1']
#    #roi_prob = out['conv6-2']
#    #pts_prob = out['conv6-3']
#    roi_prob = out['fc6-2']
#    pts_prob = out['fc6-3']
#    roi_prob = roi_prob - roi_prob
#    print(roi_prob)
#    print(cls_prob.shape)
#    print(len(rectangles))
#    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
#    rectangles = tools.NMS(rectangles,0.7,'iou')
#    print(len(rectangles))
    return rectangles

threshold = [0.8,0.6,0.8]
#imgpath = "../prepare_data/WIDER_train/images/0--Parade/0_Parade_marchingband_1_131.jpg"
#imgpath = "../prepare_data/WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_102.jpg"
#imgpath = "../prepare_data/WIDER_val/images/0--Parade/0_Parade_marchingband_1_765.jpg"
#imgpath = "../prepare_data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_71.jpg"
#imgpath = "../prepare_data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_441.jpg"
#imgpath = "../prepare_data/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg"
#imgpath = "../prepare_data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_102.jpg"
#imgpath = "../prepare_data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_865.jpg"
imgpath = "../prepare_data/WIDER_val/images/49--Greeting/49_Greeting_peoplegreeting_49_153.jpg"

rectangles = detectFace(imgpath,threshold)
img = cv2.imread(imgpath)
draw = img.copy()
print(len(rectangles))
for rectangle in rectangles:
    cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
#    cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
#    for i in range(5,15,2):
#            cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
#cv2.imshow("test",draw)
cv2.waitKey()
cv2.imwrite('test.jpg',draw)

