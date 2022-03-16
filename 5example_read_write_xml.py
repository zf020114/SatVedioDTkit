# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import DataFunction
import cv2
import shutil

data_dir='/media/zf/E/Dataset/ICPR Dataset/annotations/val_xml_gt'#'/home/zf/dataset/2019Tianzhi/ship_aug4.0/'
ext_list='.xml'
output_dir = '/media/zf/E/Dataset/ICPR Dataset/annotations/val_xml_result_example'

NAME_LABEL_MAP = {
    'airplane':       1,
    'car':       2,
    'ship': 3,
    'train': 4,
    }
def get_label_name_map(NAME_LABEL_MAP):
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEL_NAME_MAP = get_label_name_map(NAME_LABEL_MAP)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

images_list =  DataFunction.get_file_paths_recursive(data_dir,ext_list)
images_list =sorted(images_list)
total_num=len(images_list)

for i_index, image_path in enumerate(images_list):
    img_height, img_width, box_list = DataFunction.read_VOC_xml(image_path,NAME_LABEL_MAP)
    #box_list:   [xmin,ymin,xmax,ymax,label]
    if len(box_list)>0:
        box_list = np.array(box_list)
        box_list = box_list[0:2,:]
        #直接读取 GTxml 没有置信度 这里模拟检测文件的置信度
        scores = 0.5*np.ones_like(box_list)[:,0:1]
        box_list = np.hstack((box_list,scores))
        box_list=box_list.tolist()

    DataFunction.write_result_VOC_xml(output_dir,image_path,[img_height,img_width,3],1.0,'satvediodt',box_list,LABEL_NAME_MAP)
