import glob
import os
import json
import cv2
import random
import time
from PIL import Image

work_dir = '/home/user/weix/simulation_datasets/'
coco_format_save_path = work_dir + 'coco/'                      #要生成的标准coco格式标签所在文件夹
yolo_format_classes_path = work_dir + 'class.names'     #类别文件，一行一个类
yolo_format_annotation_path= work_dir + 'labels/'        #yolo格式标签所在文件夹
img_pathDir = work_dir + 'images/'                        #图片所在文件夹

with open(yolo_format_classes_path,'r') as fr:                               #打开并读取类别文件
    lines1=fr.readlines()
# print(lines1)
categories=[]                                                                 #存储类别的列表
for j,label in enumerate(lines1):
    label=label.strip()
    categories.append({'id':j+1,'name':label,'supercategory':'None'})         #将类别信息添加到categories中
# print(categories)

write_json_context=dict()                                                      #写入.json文件的大字典
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': '2021-07-25'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

#接下来的代码主要添加'images'和'annotations'的key值
imgs = glob.glob(img_pathDir+'*.png')                                        #遍历该文件夹下的所有文件，并将所有文件名添加到列表中
for i in range(3000,4000):
    # imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径

    imagePath = imgs[i]
    image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
    W, H = image.size
    # W = 2560
    # H = 1600
    img_context={}                                                              #使用一个字典存储该图片信息
    img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    img_context['file_name']=img_name
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2021-07-25'
    img_context['id']=i                                                         #该图片的id
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)                            #将该图片信息添加到'image'列表中


    txtFile=img_name.replace('.png','.txt')                                            #获取该图片获取的txt文件
    with open(os.path.join(yolo_format_annotation_path,txtFile),'r') as fr:
        lines=fr.readlines()                                                   #读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
    for j,line in enumerate(lines):

        bbox_dict = {}                                                          #将每一个bounding box信息存储在该字典中
        # line = line.strip().split()
        # print(line.strip().split(' '))
        # print(line,txtFile)
        anns = line.strip().split(' ')
        # print(len(anns))
        boxs_num = (len(anns)-1)/4
        #获取每一个标注框的详细信息
        class_id = int(anns[0])
        for n in range(int(boxs_num)):
            x, y, w, h = float(anns[1+4*n]), float(anns[2+4*n]), float(anns[3+4*n]), float(anns[4+4*n])       #将字符串类型转为可计算的int和float类型

            xmin=(x-w/2)*W                                                                    #坐标转换
            ymin=(y-h/2)*H
            xmax=(x+w/2)*W
            ymax=(y+h/2)*H
            w=w*W
            h=h*H

            bbox_dict['id']=i*100+j*10+n                                                         #bounding box的坐标信息
            bbox_dict['image_id']=i
            bbox_dict['category_id']=class_id+1                                               #注意目标类别要加一
            bbox_dict['iscrowd']=0
            height,width=abs(ymax-ymin),abs(xmax-xmin)
            bbox_dict['area']=height*width
            bbox_dict['bbox']=[xmin,ymin,w,h]
            bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
            write_json_context['annotations'].append(bbox_dict)                               #将每一个由字典存储的bounding box信息添加到'annotations'列表中

name = os.path.join(coco_format_save_path,"test"+ '.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(write_json_context,fw,indent=2)
