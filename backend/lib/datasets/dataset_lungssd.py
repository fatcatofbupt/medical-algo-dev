from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import os
import glob
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from .factory import DatasetFactory
from PIL import Image
import json
import cv2
import numpy as np
import shutil
# 图片的预处理所需要的
class BaseTransform:
    def __init__(self, size, mean = (104, 117, 123)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
    def base_transform(self,image, size, mean):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x
    def __call__(self, image):
        return self.base_transform(image, self.size, self.mean)

# Label map
# Label map
voc_labels = ('background','nodules')
label_map = {k: v for v, k in enumerate(voc_labels)}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping



def parse_annotation(annotation_path:str):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        label = 1
        # TODO: 这边只能暂时这样写
        # 这边不区分类别，因此找到的都是肺结节
        # if label not in label_map:
        #     continue
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        labels.append(label)
        boxes.append([xmin, ymin, xmax, ymax])
        difficulties.append(difficult)
    return {'boxes': boxes,'labels':labels,'difficults':difficulties}
     
@DatasetFactory.register('dataset_lungssd_basic')
class LungSegBasic(Dataset):
    params = {
        "root_dir":{
            "description":"模型在obs上的根路径",
            "value_type":"string",
            "value_range":"正确路径即可",
            "default":"test/lung_seg"
        },
        "resize":{
            "description":"图像缩放的边长",
            "value_type":"int",
            "value_range":"要与模型支持的输入大小一致",
            "default":300
        }
    }
    def __init__(self, root_dir = 'datasets/test_dataset_lung_ssd',resize = 300):
        self.root_dir = root_dir
        self.img_path = os.path.join(root_dir,'image')
        self.xml_path = os.path.join(root_dir,'Annotations')
        self.resize = resize
        self.keep_difficult = False
        # Read data files
        # 存储照片名称 
        self.image_name_list = os.listdir(self.img_path)
        self.image_name_list.sort()
        # 存储标注名称
        self.annotations_name_list = os.listdir(self.xml_path)
        self.annotations_name_list.sort()
        # 图片处理类
        self.transform =   BaseTransform(self.resize)

    def __getitem__(self, index):
        """Get a sample pair (image,mask) by an index"""
        # Read image
        image = cv2.imread(os.path.join(self.img_path,self.image_name_list[index]))
        # Read annotation
        objects = parse_annotation(os.path.join(self.xml_path,self.annotations_name_list[index]))
        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)          
        labels = torch.IntTensor(objects['labels'])
        difficults = torch.IntTensor(objects['difficults'])
        # record the meta information
        meta = {'path': self.image_name_list[index],'root_path':self.root_dir}

        # record the boxes and labels information
        new_objects = {'boxes':boxes,'labels':labels,'difficults':difficults}
        
        return (image, new_objects, meta)

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.image_name_list) 

    @classmethod
    def init(cls, init_params):
        print("Dataset Init!")
        root_dir_target = init_params['root_dir']
        root_dir_target = Path(root_dir_target)
        root_dir_target.mkdir(parents=True,exist_ok=True)
        (root_dir_target/'image').mkdir(parents=True,exist_ok=True)
        (root_dir_target/'Annotations').mkdir(parents=True,exist_ok=True)
        dataset_params = {
            'root_dir': str(root_dir_target),
        }
        return dataset_params

    
    @classmethod
    # TODO:这地方还没写完，因为暂时流程没有到这里，先搁置
    def add(cls, img_new, mask_new, meta_new, dataset_info):
        dataset_info = dataset_info.copy() # 可以不用
        new_root_dir = dataset_info['root_dir']
        new_root_dir = Path(new_root_dir)
        image_path_str = meta_new['path']
        image_output_path = new_root_dir/'image'/os.path.basename(image_path_str)
        image_output_path.parent.mkdir(parents=True,exist_ok=True)
        data_output_path = new_root_dir/'detect_results'
        data_output_path.mkdir(parents=True,exist_ok=True)
        xml_output_path = new_root_dir/'Annotations'
        xml_output_path.mkdir(parents=True,exist_ok=True)
        # 保存原始标注信息
        shutil.copyfile(os.path.join(meta_new['root_path'],'Annotations',meta_new['path'].replace('jpg','xml'))
                        ,os.path.join(str(xml_output_path),meta_new['path'].replace('jpg','xml')))
        # 保存图片
        cv2.imwrite(str(image_output_path),img_new,[cv2.IMWRITE_JPEG_QUALITY,100])
        # 记录新结果
        filename = os.path.join(data_output_path
                        ,os.path.basename(image_path_str).replace('jpg','json'))
        if(mask_new['attacked_scores'] == []):
            data_input = {'attacked_image_path':str(image_output_path),
                        'boxes_new':mask_new['attacked_boxes'],
                        'scores_new':mask_new['attacked_scores']}
        else:
            data_input = {'attacked_image_path':str(image_output_path),
                        'boxes_new':mask_new['attacked_boxes'],
                        'scores_new':mask_new['attacked_scores'][0].detach().cpu().numpy().tolist()}
        
        with open(filename, 'w') as j:
            json.dump(data_input, j)
        return dataset_info

