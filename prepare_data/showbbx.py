import cv2
import os

#wild_root = '/home/lianhe/WIDER_val/images'
wild_root = '../prepare_data/WIDER_val/images'
anno_txt = 'wider_face_train_bbx_gt.txt'

last_file=""
last_rect=[]
with open(anno_txt,"r") as f:
    for each in f.readlines():
        each = each.rstrip()
        if each.find("jpg")!=-1:

          if each != last_file and last_file != "":
              image_path = os.path.join(wild_root,last_file)
#              print image_path
              img = cv2.imread(image_path,1)
#              print image_path
#              print img
              for rect in last_rect:
                  print rect
                  [x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose] = rect
                  #print (x1,y1),(x1+w,y1+h)
                  cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
              cv2.imshow("img",img)
              cv2.waitKey(0)
#              break;
          last_file = each
          last_rect = []
        else:
          try:
            mlist = each.split()
            if len(mlist) > 2:
              mlist = [int(i) for i in mlist]
              #print mlist
              last_rect.append(mlist)
          except:
            pass
