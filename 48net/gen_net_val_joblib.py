import numpy as np
import numpy.random as npr
import joblib as pickle

net_size = 48
net = str(net_size)
with open('%s_val/pos_%s.txt'%(net, net_size), 'r') as f:
    pos = f.readlines()
#    size = round(len(pos) * 0.3)
    pos = pos[:3000]


with open('%s_val/neg_%s.txt'%(net, net_size), 'r') as f:
    neg = f.readlines()
    size = round(len(neg) * 0.3)
    #pos = pos[:3000]
    neg = neg[:6000]

with open('%s_val/part_%s.txt'%(net, net_size), 'r') as f:
    part = f.readlines()
    size = round(len(part) * 0.3)
    part = part[:3000]
    
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
    
import sys
import cv2
import os
import numpy as np

cls_list = []
roi_list = []

print('\n'+'positive-%d' % net_size)
cur_ = 0
sum_ = len(pos)
for line in pos:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=net_size or w!=net_size:
        im = cv2.resize(im,(net_size,net_size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = 1
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi])

    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])


print('\n'+'negative-%d' % net_size)
cur_ = 0
neg_keep = npr.choice(len(neg), size=6000, replace=False)
#neg_keep = npr.choice(len(neg), size=3000, replace=False)
sum_ = len(neg_keep)
for i in neg_keep:
    line = neg[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=net_size or w!=net_size:
        im = cv2.resize(im,(net_size,net_size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = 0
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi]) 
#import _pickle as pickle
fid = open("%d_val/cls.imdb" % net_size,'wb')
pickle.dump(cls_list, fid)
fid.close()

print('\n'+'part-%d' % net_size)
cur_ = 0
part_keep = npr.choice(len(part), size=3000, replace=False)
#part_keep = npr.choice(len(part), size=3000, replace=False)
sum_ = len(part_keep)
for i in part_keep:
    try:
        line = part[i]
        view_bar(cur_,sum_)
        cur_ += 1
        words = line.split()
        image_file_name = words[0]+'.jpg'
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        if h!=net_size or w!=net_size:
            im = cv2.resize(im,(net_size,net_size))
        im = np.swapaxes(im, 0, 2)
        im = (im - 127.5)/127.5
        label    = -1
        roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
        pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        roi_list.append([im,label,roi])
    except:
        pass

fid = open("%d_val/roi.imdb" % net_size,'wb')
pickle.dump(roi_list, fid)
fid.close()
