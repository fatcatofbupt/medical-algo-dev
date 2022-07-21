import pathlib
from torch._C import parse_type_comment
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import cv2

import torch
from torch.utils.data import Dataset
import skimage.io
import json
from pathlib import Path
from .factory import DatasetFactory


@DatasetFactory.register('dataset_lungseg_basic')
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
            "default":864
        }
    }
    def __init__(self, root_dir = './',resize = 864):
        self.img_path = os.path.join(root_dir,'images')
        self.mask_path = os.path.join(root_dir,'masks')
        self.list = os.listdir(self.img_path)
        self.resize = resize
        
    def __getitem__(self, index):
        """Get a sample pair (image,mask) by an index"""
        image = cv2.imread(os.path.join(self.img_path, self.list[index]))
        mask = cv2.imread(os.path.join(self.mask_path, self.list[index]))
        print("Image Path",os.path.join(self.img_path, self.list[index]))
        image = cv2.resize(image, (self.resize, self.resize))
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.resize, self.resize))
        meta = {'path': self.list[index]}
        return (image, mask, meta)

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.list) 

    @classmethod
    def init(cls, init_params):
        print("Dataset Init!")
        root_dir_target = init_params['root_dir']
        root_dir_target = Path(root_dir_target)
        root_dir_target.mkdir(parents=True,exist_ok=True)
        (root_dir_target/'images').mkdir(parents=True,exist_ok=True)
        (root_dir_target/'masks').mkdir(parents=True,exist_ok=True)
        dataset_params = {
            'root_dir': str(root_dir_target),
        }
        return dataset_params

    
    @classmethod
    def add(cls, img_new, mask_new, meta_new, dataset_info):
        dataset_info = dataset_info.copy() # 可以不用
        root_dir = dataset_info['root_dir']
        root_dir = Path(root_dir)
        image_path_str = meta_new['path']
        image_output_path = root_dir/'images'/image_path_str
        image_output_path.parent.mkdir(parents=True,exist_ok=True)
        mask_output_path = root_dir/'masks'/image_path_str
        mask_output_path.parent.mkdir(parents=True,exist_ok=True)
        # 保存新图片和掩模
        if isinstance(img_new,torch.Tensor):
            img_new = img_new.to('cpu').numpy()
            mask_new = mask_new.to('cpu').numpy()
        skimage.io.imsave(fname=image_output_path,arr=img_new)
        skimage.io.imsave(fname=mask_output_path,arr=mask_new)
        # dataset_info['_count'] = count+1
        return dataset_info
