# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import DataFunction
import cv2

data_dir='/media/zf/E/Dataset/ICPR Dataset/validation data'
json_name = os.path.join(os.path.dirname(data_dir),'instances_test2017.json')
ext_list='.jpg'
Rotatexmls = data_dir#os.path.join(data_dir,'val2017rotatexml')
images_path =data_dir

NAME_LABEL_MAP = {
    'airplane':       1,
    'car':       2,
    'ship': 3,
    'train': 4,
    }
def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()
voc_clses=[]
for name, label in NAME_LABEL_MAP.items():
    voc_clses.append(name)


categories = []
for iind, cat in enumerate(voc_clses):
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind + 1
    categories.append(cate)
   
def xml2bbox_seg(xmlname, rotate_box_list,img_id):
    #通过这个函数将旋转矩形框换正框转换为coco可以读取的格式。其中将旋转矩形框转换维恩带有头部的分割点集
    bbox=[]
    for ind, rectbox in enumerate(rotate_box_list):
        rotatebox=rotate_box_list[ind]
        [x_center,y_center,w,h,angle,label]=rotatebox[0:6]
        cv_rotete_rect=DataFunction.rotate_rect2cv(rotatebox[0:5])
        rect_box =cv2.boxPoints(cv_rotete_rect)
        rect_box=np.array(rect_box)
        xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])
        xmin,ymin,xmax,ymax = np.float64(xmin),np.float64(ymin),np.float64(xmax),np.float64(ymax)
        area=h*w*3/4
        RotateMatrix=np.array([
                              [np.cos(angle),-np.sin(angle)],
                              [np.sin(angle),np.cos(angle)]])

        r1,r2,r3,r4=np.transpose([-w/2,-h/2]),np.transpose([w/2,-h/2]),np.transpose([w/2,h/2]),np.transpose([-w/2,h/2])
        p1=np.transpose(np.dot(RotateMatrix, r1))+[x_center,y_center]
        p2=np.transpose(np.dot(RotateMatrix, r2))+[x_center,y_center]
        p3=np.transpose(np.dot(RotateMatrix, r3))+[x_center,y_center]
        p4=np.transpose(np.dot(RotateMatrix, r4))+[x_center,y_center]
        area=h*w*3/4
        
        bbox.append([p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1],xmin,ymin,xmax - xmin,ymax - ymin,img_id,label,area])
        # bbox.append([xmin,ymin,xmax - xmin,ymax - ymin,img_id,label,area])
    return bbox

if not os.path.isdir(os.path.split(json_name)[0]):
    os.makedirs(os.path.split(json_name)[0])

images_list =  DataFunction.get_file_paths_recursive_ICPR(images_path,ext_list)
images_list =sorted(images_list)
images=[]
bboxes = []
ann_js = {}
total_num=len(images_list)
for i_index, image_path in enumerate(images_list):
    image = {}
    vedio_index = os.path.dirname(os.path.split(image_path)[0])
    frame_index = os.path.split(image_path)[-1]
    vedio_index = os.path.basename(vedio_index)
    image['file_name']  = vedio_index+'_'+frame_index
    xml_file = image_path.replace('img1','xml').replace('.jpg','.xml')
#    xml_file=os.path.splitext(image_path)[0].replace('val2017','rotatexml_val')+'.xml'
    rotate_box_list=DataFunction.read_rec_to_rot(xml_file,NAME_LABEL_MAP)
    img = cv2.imread(image_path)
    img_shape = img.shape
    image['width'] = img_shape[1]#img_size[1]#width#
    image['height'] =img_shape[0]#img_size[0]#600#
    image['id'] =i_index#int(os.path.split(xml_file)[1].replace('.xml','').split('/')[-1]) 
    sig_xml_bbox=xml2bbox_seg(xml_file,rotate_box_list,i_index)
    images.append(image)
    bboxes.extend(sig_xml_bbox)
    if i_index%1000==0:
        print('{}/{}'.format(i_index,total_num))

ann_js['images'] = images
ann_js['categories'] = categories
annotations = []
total_box=len(bboxes)
for box_ind, box in enumerate(bboxes):
    anno = {}
    anno['image_id'] =  box[-3]
    anno['category_id'] = box[-2]
    anno['bbox'] = box[-7:-3]
    anno['id'] = box_ind
    anno['area'] = box[-1]
    anno['iscrowd'] = 0
    anno['segmentation']=[box[0:8]]#12
    annotations.append(anno)
    if box_ind%1000==0:
        print('{}/{}'.format(box_ind,total_box))
ann_js['annotations'] = annotations
json.dump(ann_js, open(json_name, 'w'), indent=4)  # indent=4 更加美观显示
print('down!')
