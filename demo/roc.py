#########################################################################################
### descript:loader imdb for testing data
###
#########################################################################################
class BatchLoader(object):
    def __init__(self,cls_list,roi_list,pts_list,net_side,cls_root,roi_root,pts_root):
        self.mean = 128
        self.im_shape = net_side
        self.cls_root = cls_root
        self.roi_root = roi_root
        self.pts_root = pts_root
        self.roi_list = []
        self.cls_list = []
        self.pts_list = []
        print("Start Reading Classify Data into Memory...")
        if imdb_exit:
            fid = open('12_test/cls.imdb','rb')
            self.cls_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(cls_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.cls_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = int(words[1])
                roi      = [-1,-1,-1,-1]
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.cls_list.append([im,label,roi,pts])
        random.shuffle(self.cls_list)
        self.cls_cur = 0
        print("\n",str(len(self.cls_list))," Classify Data have been read into Memory...")

        print("Start Reading Regression Data into Memory...")
        if imdb_exit:
            self.roi_list = [] 
            #fid = open('12_test/roi.imdb','rb')
            #self.roi_list = pickle.load(fid)
            #fid.close()
        else:
            fid = open(roi_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.roi_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = int(words[1])
                roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.roi_list.append([im,label,roi,pts])
        random.shuffle(self.roi_list)
        self.roi_cur = 0 
        print("\n",str(len(self.roi_list))," Regression Data have been read into Memory...")

        print( "Start Reading pts-regression Data into Memory...")
        if imdb_exit:
            self.pts_list = []
            #fid = open('12/pts.imdb','rb')
            #self.pts_list = pickle.load(fid)
            #fid.close()
        else:
            fid = open(pts_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.pts_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = -1
                roi      = [-1,-1,-1,-1]
                pts         = [float(words[ 6]),float(words[ 7]),
                            float(words[ 8]),float(words[ 9]),
                            float(words[10]),float(words[11]),
                            float(words[12]),float(words[13]),
                            float(words[14]),float(words[15])]
                self.pts_list.append([im,label,roi,pts])
        random.shuffle(self.pts_list)
        self.pts_cur = 0 
        print("\n",str(len(self.pts_list))," pts-regression Data have been read into Memory...")

    def load_next_image(self,loss_task): 
        if loss_task == 0:
            if self.cls_cur == len(self.cls_list):
                self.cls_cur = 0
                random.shuffle(self.cls_list)
            while self.cls_list[self.cls_cur][1] != 1:
                self.cls_cur += 1
            cur_data = self.cls_list[self.cls_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            roi      = [-1,-1,-1,-1]
            pts             = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            if random.choice([0,1])==1:
                im = cv2.flip(im,random.choice([-1,0,1]))
            self.cls_cur += 1
            return im, label, roi, pts

        if loss_task == 1:
            if self.roi_cur == len(self.roi_list):
                self.roi_cur = 0
                random.shuffle(self.roi_list)
            cur_data = self.roi_list[self.roi_cur]  # Get the image index
            im             = cur_data[0]
            label    = -1
            roi      = cur_data[2]
            pts             = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.roi_cur += 1
            return im, label, roi, pts

        if loss_task == 2:
            if self.pts_cur == len(self.pts_list):
                self.pts_cur = 0
                random.shuffle(self.pts_list)
            cur_data = self.pts_list[self.pts_cur]  # Get the image index
            im             = cur_data[0]
            label    = -1
            roi      = [-1,-1,-1,-1]
            pts             = cur_data[3]
            self.pts_cur += 1
            return im, label, roi, pts


def main():
    deploy = '12net.prototxt'
    caffemodel = '12net.caffemodel'
    net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

    self.batch_size = 1000
    net_side = 12
    cls_list = ''
    roi_list = ''
    pts_list = ''
    cls_root = ''
    roi_root = ''
    pts_root = ''
    self.batch_loader = BatchLoader(cls_list,roi_list,pts_list,net_side,cls_root,roi_root,pts_root)
    top[0].reshape(self.batch_size, 3, net_side, net_side)
    top[1].reshape(self.batch_size, 1)
    top[2].reshape(self.batch_size, 4)
    top[3].reshape(self.batch_size, 10)


    #loss_task = random.randint(0,2)
    #loss_task = random.randint(0,1)
    loss_task = 0
    for itt in range(self.batch_size):
        im, label, roi, pts= self.batch_loader.load_next_image(loss_task)
        net_12.blobs['data'].reshape(1,3,12,12)
        caffe.set_device(0)
        caffe.set_mode_gpu()
        out_ = net_12.forward()
        prob =  net_12['prob1'][0][1]
        print(label,prob)

if __name__ == "__main__":
    main()
