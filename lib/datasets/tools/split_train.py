import os.path
from glob import glob
work_dir = '/data/liuweixing/UAVDT/dataset/Annotations/'
labels = glob(work_dir + '*.xml')
fog_txt = open('/data/liuweixing/UAVDT/dataset/UAVDT_fog_VOC/ImageSets/Main/trainval.txt','w')
uavdt_txt = open('/data/liuweixing/UAVDT/dataset/UAVDT_VOC/ImageSets/Main/trainval.txt','w')
for label in labels:
    name = os.path.basename(label)[:-4]
    if 'M0704' in label or 'M0701' in label or 'M0501' in label:
        fog_txt.write(name)
        fog_txt.write('\n')
    else:
        uavdt_txt.write(name)
        uavdt_txt.write('\n')

