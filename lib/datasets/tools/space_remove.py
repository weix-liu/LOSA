import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC
import os

import tqdm
from PIL import Image

# clsname_nums = ['airplane','ship','storagetank','baseballfield','tenniscourt','basketballcourt','groundtrackfield','harbor','bridge','vehicle']
# _classes = ['airplane', 'ship', 'storagetank', 'baseballfield','tenniscourt', 'basketballcourt', 'groundtrackfield', 'harbor', 'bridge', 'vehicle']
clsname_nums = 'plane'
#
# _classes =  []
def parse_xml(xml_path):
    cls_count = 0
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')

    for ix, obj in enumerate(objs):
        name = obj.find('name').text

        if name == clsname_nums:
            cls_count = cls_count + 1

    if cls_count == 0:
        print(xml_path)
    print(cls_count)
    return cls_count

#
#
# xmls = glob.glob('C:\\Users\\19331\\Documents\\dataset\\vedai\\VEDAIIR\\Annotations\\*')
# # xmls = glob.glob('C:\\Users\\19331\\Documents\\dataset\\NWPUVHR-10\\voc\\*')
# for xml in tqdm.tqdm(xmls):
#     parse_xml(xml)
# print(clsname_nums)

work_dir = 'C:\\Users\\19331\\Documents\\dataset\\xview\\XviewVOC\\'
traintxt = work_dir + 'ImageSets\\Main\\trainval.txt'
plane_traintxt = work_dir + 'ImageSets\\Main\\plane_trainval.txt'
xmldir = work_dir + 'Annotations\\'

out = []
f2 = open(plane_traintxt,'w')
with open(traintxt,'r') as f:
    namelist = f.readlines()
    for name in namelist:
        xmlfile = xmldir + name.strip() + '.xml'
        if parse_xml(xmlfile)>0:
            png = name.strip()
            f2.write(png)
            f2.write('\n')


#
# traintxt = work_dir + 'ImageSets\\Main\\trainval.txt'
# with open(traintxt,'r') as f:
#     namelist = f.readlines()
#     for name in namelist:
#         xmlfile = xmldir + name.strip() + '.xml'
# #         parse_xml(xmlfile)
# # print(clsname_nums)
#         if parse_xml(xmlfile)>0:
#             png = name.strip()
#             out.append(png)
#
# print(len(out))
# print(clsname_nums)



# dir = '/data/liuweixing/NWPU/ImageSets/Main/'
# alltxt = dir + 'ALLtrainval.txt'
# traintxt = dir + 'trainval.txt'
# testtxt = dir + 'test.txt'
# import random
# a = []
# for i in range(650):
#     a.append(i)
# random.shuffle(a)
# print(a)
#
# with open(alltxt,'r') as f:
#     info = f.readlines()
#
# with open(traintxt,'w') as f:
#     for o in a[:325]:
#         f.write(info[o])
#         # f.write('\n')
# with open(testtxt,'w') as f:
#     for o in a[325:]:
#         f.write(info[o])
#         # f.write('\n')





# classes = ('__background__',  # always index 0
#                  'bus', 'bicycle', 'car', 'motorcycle', 'pedestrian', 'trafficsign', 'trafficlight', 'rider', 'train',
#                  'truck')
# class_to_ind = dict(zip(classes, range(len(classes))))
# print(class_to_ind['trafficsign'])