import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import cv2
import os
_classes = {'__background__':0,  # always index 0
              'car': 1, 'fcar': 2, 'bus': 3, 'truck': 4, 'van': 5}
#_classes = {'__background__':0,  # always index 0
#            'airplane':1}

import numpy as np
np.random.seed(2024)
colors = np.random.uniform(0, 255, size=(10, 3))
color = (0, 255, 0)
workdir = 'C:\\Users\\19331\\Documents\\dataset\\dronevehicle\\test\\inf\\'
imgsets = workdir + 'test.txt'
img_dir = workdir + 'JPEGImages\\'
save_dir = workdir + 'vis_res\\GT\\'
ann_dir = workdir + 'Annotations\\'

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    newboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        # newboxes.append(name)

        # if name == 'car':
        bbox = obj.find('bndbox')
        box = []
        box.append(float(bbox.find('xmin').text))
        box.append(float(bbox.find('ymin').text))
        box.append(float(bbox.find('xmax').text))
        box.append(float(bbox.find('ymax').text))
        box.append(name)
        newboxes.append(box)
    return newboxes

def vis_box(imgfile,boxs):
    img = cv2.imread(imgfile)
    for box in boxs:
        # score = box[-1]
        # if score<0.5:
        #     continue
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])
        name = box[-1]
        # name = 'vehicle'
        #print(name)
        c = int(_classes[name])-1

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[c], 2)
        y = ymin - 5 if ymin - 5 > 5 else ymin + 5
        cv2.putText(img, '%s' % (name), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[c], 2)

    cv2.imwrite(save_dir+os.path.basename(imgfile),img)


if __name__ == '__main__':
    # se_list = ['1127_2417_1800','1154_0000_1800','2122_2210_1800','2523_0000_2313','2489_1800_0000']
    # se_list = ['P0060']
    with open(imgsets,'r') as f:
        imgs = f.readlines()
        # for i in se_list:
        for i in range(500,550):
        # for i in range(1000,1050):
            img0 = img_dir + imgs[i].strip() + '.jpg'
            bboxs = parse_xml(ann_dir + imgs[i].strip() + '.xml')
        #     img0 = img_dir + i +'.jpg'
        #     bboxs = parse_xml(ann_dir + i.strip() + '.xml')
            vis_box(img0,bboxs)
