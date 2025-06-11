import os
import numpy as np
from itertools import groupby
from skimage import morphology,measure
import imageio
import cv2
# 因为一张图片里只有一种类别的目标，所以label图标记只有黑白两色
rgbmask = np.array([[0,0,0],[255,255,255]],dtype=np.uint8)

# 从label图得到 boundingbox 和图上连通域数量 object_num
def getboundingbox(mask_image):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
    mask[np.where(np.all(mask_image == rgbmask[1],axis=-1))[:2]] = 1
    # 删掉小于10像素的目标
    mask_without_small = morphology.remove_small_objects(mask,min_size=10,connectivity=2)
    # 连通域标记
    label_image = measure.label(mask_without_small)
    #统计object个数
    object_num = len(measure.regionprops(label_image))
    boundingbox = list()
    for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
        boundingbox.append(region.bbox)
    return object_num, boundingbox

import xml.etree.ElementTree as ET

def createXMLlabel(savedir,objectnum, bbox, classname, foldername='0',filename='0', path='0', database='road', width='400', height='600',depth='3', segmented='0', pose="Unspecified", truncated='0', difficult='0'):
    # 创建根节点
    root = ET.Element("annotation")

    # 创建子节点
    folder_node = ET.Element("folder")
    folder_node.text = foldername
    # 将子节点数据添加到根节点
    root.append(folder_node)

    file_node = ET.Element("filename")
    file_node.text = filename
    root.append(file_node)
    path_node = ET.Element("path")
    path_node.text = path
    root.append(path_node)

    source_node = ET.Element("source")
    # 也可以使用SubElement直接添加子节点
    db_node = ET.SubElement(source_node, "database")
    db_node.text = database
    root.append(source_node)

    size_node = ET.Element("size")
    width_node = ET.SubElement(size_node, "width")
    height_node = ET.SubElement(size_node, "height")
    depth_node = ET.SubElement(size_node, "depth")
    width_node.text = width
    height_node.text = height
    depth_node.text = depth
    root.append(size_node)

    seg_node = ET.Element("segmented")
    seg_node.text = segmented
    root.append(seg_node)

    for i in range(objectnum):
        newEle = ET.Element("object")
        name = ET.Element("name")
        name.text = classname
        newEle.append(name)
        pose_node = ET.Element("pose")
        pose_node.text = pose
        newEle.append(pose_node)
        trunc = ET.Element("truncated")
        trunc.text = truncated
        newEle.append(trunc)
        dif = ET.Element("difficult")
        dif.text = difficult
        newEle.append(dif)
        boundingbox = ET.Element("bndbox")
        xmin = ET.SubElement(boundingbox, "xmin")
        ymin = ET.SubElement(boundingbox, "ymin")
        xmax = ET.SubElement(boundingbox, "xmax")
        ymax = ET.SubElement(boundingbox, "ymax")
        xmin.text = str(bbox[i][1])
        ymin.text = str(bbox[i][0])
        xmax.text = str(bbox[i][3])
        ymax.text = str(bbox[i][2])
        newEle.append(boundingbox)
        root.append(newEle)

    ImageID = filename.split('.')[0]
    # 创建elementtree对象，写入文件
    tree = ET.ElementTree(root)
    tree.write(savedir + '/'+ ImageID + ".xml")

if __name__ == '__main__':
    imagedir = r'C:\迅雷下载\Building change detection dataset_add\1. The two-period image data\2012\splited_images\train\label'
    saveXMLdir = r'C:\迅雷下载\Building change detection dataset_add\1. The two-period image data\2012\splited_images\train\Annotations'

    if os.path.exists(saveXMLdir) is False:
        os.mkdir(saveXMLdir)

    for root, _, fnames in sorted(os.walk(imagedir)):
        for fname in sorted(fnames):
            labelpath = os.path.join(root, fname)
            labelimage = imageio.imread(labelpath)
            labelimage = np.expand_dims(labelimage,-1)
            labelimage = np.concatenate((labelimage,labelimage,labelimage),axis=-1)
            # 得到label图上的boundingingbox和数量
            objectnum, bbox = getboundingbox(labelimage)

            labelfilename = os.path.basename(labelpath)
            classname = 'building'
            origin_image_name = labelfilename[:-4] + '.tif'

            # 一些图片信息
            foldername = 'whu2012_dataset'
            path = origin_image_name
            database = 'whu2012_dataset'
            width = str(labelimage.shape[0])
            height = str(labelimage.shape[1])
            depth = str(labelimage.shape[2])

            createXMLlabel(saveXMLdir, objectnum, bbox, classname, foldername=foldername, filename=origin_image_name,
                           path=path,
                           database=database, width=width, height=height, depth=depth, segmented='0',
                           pose="Unspecified",
                           truncated='0', difficult='0')