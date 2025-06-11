import os
import pdb

import numpy as np
import codecs
import json
from glob import glob
import cv2


# workdir = 'C:\\Users\\19331\\Documents\\dataset\\xview\\train_labels\\'
# f = open(workdir+'xView_train.geojson','r')
# data = json.load(f)
#
# img_label = {}
#
# n = 0
# for i in range(len(data['features'])):
# # for i in range(100):
#     if data['features'][i]['properties']['bounds_imcoords'] != []:
#         b_id = data['features'][i]['properties']['image_id']
#         cls_id = data['features'][i]['properties']['type_id']
#         cls = None
#         if cls_id in [11,12,13,15]:
#             # cls = 'plane'
#             cls = 1
#         if cls_id in [40,41,42,44,45,47,49,50,51,52]:
#             # cls = 'ship'
#             cls = 2
#         if cls_id in [86]:
#             # cls = 'storage'
#             cls = 3
#             n = n + 1
#
#         if cls is not None:
#             val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
#             if val.shape[0] != 4:
#                 print("Issues at %d!" % i)
#             else:
#                 box = val.tolist()
#                 box.append(cls)
#                 if b_id in img_label.keys():
#                     img_label[b_id].append(box)
#                 else:
#                     a = []
#                     a.append(box)
#                     img_label[b_id] = a
# print(n)
# f2 = open(workdir+'xview3cls.json','w')
# json.dump(img_label,f2)

# train_files = 'C:\\Users\\19331\\Documents\\dataset\\AIRSAR\\test.txt'
# train_list = []
# with open(train_files) as f:
#     trainnames = f.readlines()
#     for name in trainnames:
#         name = name.split('.tif')[0]
#         train_list.append(name)
#
# imgs = glob('C:\\Users\\19331\\Documents\\dataset\\xview\\VOC\\Annotations\\*')
# with open('C:\\Users\\19331\\Documents\\dataset\\xview\\VOC\\trainval.txt','w') as f:
#     for img in imgs:
#
#         f.write(os.path.basename(img).split('.xm')[0])
#         f.write('\n')


clsid_name = [
    'bridge',
    'airplane',
    'groundtrackfield',
    'vehicle',
    'parkinglot',
    'Tjunction',
    'baseballdiamond',
    'tenniscourt',
    'basketballcourt',
    'ship',
    'crossroad',
    'harbor',
    'storagetank'
]

json_dir = "C:\\Users\\19331\\Documents\\dataset\\HRRSD\\"  # 原始labelme标注数据路径
saved_path = "C:\\Users\\19331\\Documents\\dataset\\HRRSD\\xmls\\"  # 保存路径

json_filename = json_dir + "hrrsd_val_m-fld_4352_3084.json"
json_file = json.load(open(json_filename, "r", encoding="utf-8"))
imgs = json_file['images']
bboxs = json_file['annotations']
with open('C:\\Users\\19331\\Documents\\dataset\\HRRSD\\test.txt','w') as f:
    for img in imgs:
        name = img['file_name'].split('.')[0]
        f.write(name)
        f.write('\n')

for img in imgs:
    name = img['file_name'].split('.')[0]
    height = img['height']
    width = img['width']
    img_id = img['id']


    with codecs.open(saved_path + name + ".xml", "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'HRRSD' + '</folder>\n')
        xml.write('\t<filename>' + name + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>HRRSD airplane</database>\n')
        xml.write('\t</source>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')

        for box in bboxs:
            if box['image_id'] == img_id:
                clsid = box['category_id']
                if clsid>12:
                    print(clsid)
                    continue
                name = clsid_name[clsid]

                rect = np.array(box["bbox"])
                xmin = int(rect[0])
                xmax = int(rect[0] + rect[2])
                ymin = int(rect[1])
                ymax = int(rect[1] + rect[3])

                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + name + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')

        xml.write('</annotation>')

