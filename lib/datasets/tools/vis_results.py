import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
# _classes = ('__background__',  # always index 0
#                          'airplane', 'ship', 'storagetank', 'baseballdiamond','tenniscourt', 'basketballcourt', 'groundtrackfield', 'harbor', 'bridge', 'vehicle')
_classes = ('__background__',  # always index 0
           'vehicle')
# _classes = ('__background__',  # always index 0
#             'car','fcar','bus','truck','van')
cls_name = {}
for i in range(len(_classes)):
    cls_name[i] = _classes[i]
color = (0, 255, 0)
workdir = 'C:\\Users\\19331\\Documents\\dataset\\CORS-ADD-HBB\\images\\'
imgsets = workdir + 'ImageSets\\Main\\test.txt'
img_dir = workdir + 'gray\\'

np.random.seed(2024)
colors = np.random.uniform(0, 255, size=(10, 3))

def vis_box(imgfile,allboxs,save_dir,img_index):
    img = cv2.imread(imgfile)
    # for c in range(5):
    for c in range(1):
        c_box = allboxs[c+1][img_index]
        name = cls_name[c+1]
        for box in c_box:
            score = box[-1]
            if score<0.5:
                continue
            xmin = int(box[0])
            xmax = int(box[2])
            ymin = int(box[1])
            ymax = int(box[3])
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[c], 2)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            y = ymin - 5 if ymin - 5 > 5 else ymin + 5
            # cv2.putText(img, '{:s}:{:.2f}'.format(name,score), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[c], 1)
            cv2.putText(img, '{:.2f}'.format(score), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  color, 1)

    # cv2.imwrite(save_dir+os.path.basename(imgfile),img)
    cv2.imencode('.jpg', img)[1].tofile(save_dir+os.path.basename(imgfile))

if __name__ == '__main__':
    save_dir = workdir + 'vis_dscr\\losa\\'
    # save_dir = 'C:\\Users\\19331\\Documents\\dataset\\CORS-ADD-HBB\\images\\vis_dscr\\ours\\'
    bboxs = np.load(save_dir +'detections.pkl', allow_pickle=True)
    with open(imgsets,'r') as f:
        imgs = f.readlines()
        for i in range(100, 150):
        # for i in range(1000,1050):
            img0 = img_dir + imgs[i].strip() + '.tif'
            vis_box(img0,bboxs,save_dir,img_index=i)




