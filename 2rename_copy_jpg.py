# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import DataFunction
import cv2
import shutil

# data_dir='/media/zf/E/Dataset/ICPR Dataset/training data'
# out_dir = '/media/zf/E/Dataset/ICPR Dataset/train2017_1'

data_dir='/media/zf/E/Dataset/ICPR Dataset/training data'
out_dir = '/media/zf/E/Dataset/ICPR Dataset/train2017_1'
ext_list='.jpg'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
images_list =  DataFunction.get_file_paths_recursive_ICPR(data_dir,ext_list)
images_list =sorted(images_list)

for i_index, image_path in enumerate(images_list):
    image = {}
    vedio_index = os.path.dirname(os.path.split(image_path)[0])
    frame_index = os.path.split(image_path)[-1]
    vedio_index = os.path.basename(vedio_index)
    image['file_name']  = vedio_index+'_'+frame_index
    new_name = os.path.join(out_dir,image['file_name'] )
    shutil.copy(image_path,new_name)



