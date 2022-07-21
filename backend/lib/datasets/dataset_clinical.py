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

import json

@DatasetFactory.register('dataset_clinical_basic')
class ClinicalBasic(Dataset):
    params = {
        "root_dir":{
            "description":"模型的根路径",
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

    def __init__(self, root_dir):
        super(ClinicalBasic, self).__init__()
        self.root_dir = root_dir
        self.list = os.listdir(self.root_dir)

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.list)

    def __getitem__(self, index):
        # 数据路径
        data_name = self.list[index]
        data_path = os.path.join(self.root_dir, data_name)

        # 加载数据和标签 统一数据项为text
        data = json.load(open(data_path, "r"))
        if "text" not in data.keys():
                data["text"] = data["note"]
                del data["note"]
        assert "label" in data.keys()
        clin_data = data['text']
        label = data['label']

        # 数据长度
        raw_length = len(clin_data)
        label_length = len(label)

        # entity,但不确定要不要用
        entity = data['entity']

        meta = {
                'path': data_path,
                'name': data_name,
                'raw_length': raw_length,  #TAG: 不一定需要
                'label_length': label_length,
                'entity': entity,
                'raw': data
                }

        return (clin_data, label, meta)


    @classmethod
    # 生成数据集的存储位置
    def init(cls, init_params):
        print("Dataset Init!")
        root_dir_target = init_params['root_dir']
        root_dir_target = Path(root_dir_target)
        root_dir_target.mkdir(parents=True, exist_ok=True)
        (root_dir_target/'json').mkdir(parents=True, exist_ok=True)
        dataset_params = {
            'root_dir': str(root_dir_target),
        }
        return dataset_params


    @classmethod
    def add(cls, clin_data_new, label_new, meta_new, dataset_info):
        dataset_info = dataset_info.copy() # 可以不用
        root_dir = dataset_info['root_dir']
        root_dir = Path(root_dir)
        clin_data_path_str = meta_new['name']
        clin_data_output_path = root_dir/clin_data_path_str
        clin_data_output_path.parent.mkdir(parents=True,exist_ok=True)
        # 保存新数据和标签
        # TAG:保存生成数据和标签
        data_input = {"raw_data": meta_new['raw_seq'], 
                        "raw_label": meta_new['raw_label'], 
                        "adv_data": meta_new['adv_seq'], 
                        "adv_label": meta_new['adv_label']}
        with open(clin_data_output_path, 'w') as j:
            json.dump(data_input, j)
        # dataset_info['_count'] = count+1

        return dataset_info