from genericpath import exists
import os
import json
from typing import Callable
from pathlib import Path

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import skimage.io
import requests
import zipfile

from .factory import DatasetFactory


#Pad an image to make it square
def pad_img(img):
    # print(len(img.shape))
    if len(img.shape) == 3:
        x,y,c = img.shape
    else:
        x,y = img.shape
        c = 3
        img = img[:,:,np.newaxis]
        img = np.repeat(img, c, axis=2)

    # print(c)
    if x > y:
        pad_len = int((x-y)/2)
        padded = np.zeros((x,x,3))
        if (x-y) % 2 == 0:
            padded[:,pad_len:x-pad_len,:] = img
        else:
            padded[:,pad_len+1:x-pad_len,:] = img
    else:
        pad_len = int((y-x)/2)
        padded = np.zeros((y,y,3))
        if (y-x) % 2 == 0:
            padded[pad_len:y-pad_len,:,:] = img
        else:
            padded[pad_len+1:y-pad_len,:,:] = img
    return padded


@DatasetFactory.register('dataset_fundus_binary')
class FundusBinary(Dataset):
    params = {
        "root_dir":{
            "description":"模型在obs上的根路径",
            "value_type":"string",
            "value_range":"正确路径即可",
            "default":"topic4/fundus"
        },
        "resize":{
            "description":"图像缩放的边长",
            "value_type":"int",
            "value_range":"要与模型支持的输入大小一致",
            "default":448
        },
        "data_len":{
            "description":"下载的数据大小",
            "value_type":"int",
            "value_range":"取值为1~数据集实际大小之间",
            "default":11
        },
        "download_dir":{
            "description":"数据下载的本地路径",
            "value_type":"string",
            "value_range":"正确的本地路径",
            "default":'./tmp'
        }
    }
    def __init__(self, root_dir, resize=None):
        super(FundusBinary, self).__init__()
        self.root_dir = root_dir
        self.imgs_dir = os.path.join(self.root_dir,'images')
        self.label_dir = os.path.join(self.root_dir,'labels.json')
        self.imgs = os.listdir(self.imgs_dir)
        self.resize = resize
        with open(self.label_dir) as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_path = os.path.join(self.imgs_dir,img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if not (self.resize is None):
            img = pad_img(img)
            img = cv2.resize(img,(self.resize,self.resize))
        # print(img.shape)
        # t = transforms.ToTensor()
        img = img.astype(np.uint8)
        label = int(self.labels[img_name])
        meta = {'path': img_name}
        return img, label, meta

    @classmethod
    def init(self,init_params):
        # raise Exception('Not implemented!!!')
        root_dir_target = init_params['root_dir']
        root_dir_target = Path(root_dir_target)
        root_dir_target.mkdir(parents=True,exist_ok=True)
        (root_dir_target/'images').mkdir(parents=True,exist_ok=True)

        labels = {}
        (root_dir_target/'labels.json').write_text(json.dumps(labels))
        dataset_params = {
            'root_dir': str(root_dir_target),
        }
        return dataset_params

    @classmethod
    def add(self, img, label, meta, dataset_params):
        # raise Exception('Not implemented!!!')
        dataset_params = dataset_params.copy()

        root_dir = dataset_params['root_dir']
        count = dataset_params.get('_count',0)
        root_dir = Path(root_dir)
        image_path_str = meta['path']
        image_output_path = root_dir/'images'/image_path_str
        image_output_path.parent.mkdir(parents=True,exist_ok=True)
        
        if isinstance(img,torch.Tensor):
            img = img.to('cpu').numpy()
        skimage.io.imsave(fname=image_output_path,arr=img)
        dataset_params['_count'] = count+1

        if not (label is None):
            label_path = root_dir/'labels.json'
            labels = json.loads(label_path.read_text())
            labels[image_path_str] = label
            label_path.write_text(json.dumps(labels))
        return dataset_params

    # def zip_and_upload(self,data_dir,zip_dir,upload_name):
    #     zip = zipfile.ZipFile(zip_dir,'w')
    #     print("Start Zipping...")
    #     shortest = len(data_dir.split('/')) + 1
    #     for root, _, files in os.walk(data_dir):
    #         for file in files:
    #             file_path = os.path.join(root,file)
    #             if 'model' in file_path:
    #                 continue
    #             if len(file_path.split('/'))> shortest:
    #                 zip.write(file_path,os.path.join(file_path.split('/')[-2],file))
    #             else:
    #                 zip.write(file_path,file)
    #     zip.close()
    #     print("Zip Finished!")
    #     print("Start Uploading...")
    #     with open(zip_dir,'rb') as fb:
    #         response = self.client.put_object(
    #             Bucket=self.bucket,
    #             Body=fb,
    #             Key='generated_dataset/' + upload_name, # 上传后的文件名称，可以包含文件路径
    #             StorageClass='Standard',
    #         )
    #     print(response['ETag'])
    #     print("Upload Finished!")


