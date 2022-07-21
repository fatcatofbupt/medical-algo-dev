from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
from pathlib import Path

from .factory import DatasetFactory

# 零填充
def zero_padding(raw_data:np.array,max_length:int) -> np.array:
    # 零填充
    # 填充方式：两边
    ## Input:
    # raw_data(ecg_data):date_type:np.array  
    # max_length
    ## Output:
    # data
    data = np.zeros((len(raw_data), max_length))
    for i in range(len(raw_data)):
        if len(raw_data[i]) >= max_length:
            data[i] = raw_data[i][:max_length]
        else:
            remainder = max_length - len(raw_data[i])
            data[i] = np.pad(raw_data[i], (int(remainder / 2), remainder - int(remainder / 2)), 'constant', constant_values=0)
    return data

@DatasetFactory.register('dataset_ecg_classifiction')
class Ecg_dataset(Dataset):
    # TODO:待补充
    params = {
    }
    def __init__(self, root_dir):
        super(Ecg_dataset, self).__init__()
        self.root_dir = root_dir
        self.data = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # 单个文件夹名称，该文件夹下包含心电数据和label
        data_name = self.data[index]
        ecg_path = os.path.join(self.root_dir,data_name,'%s_data.npy'%data_name)
        label_path  = os.path.join(self.root_dir,data_name,'%s_label.npy'%data_name)
        
        # 加载数据和标签        
        ecg_data = np.load(ecg_path,allow_pickle=True)
        label = np.load(label_path)


        # 为满足后续流程，这里需要升维
        ecg_data = ecg_data[np.newaxis]
        label = label[np.newaxis]

        # 记录一下原始的ecg数据的长度
        raw_length = len(ecg_data.squeeze())

        # 心电数据预处理
        # TODO:这里的处理方式是双边零填充，最长长度18000，先写死，后续有需要再补充
        ecg_data = zero_padding(ecg_data,18000)

        # 数据转格式：numpy-->tensor
        ecg_data = torch.from_numpy(ecg_data).unsqueeze(-2).type(torch.FloatTensor)
        label = torch.LongTensor(label)

        meta = {
                'path': ecg_path,
                'length':raw_length
               }

        return ecg_data, label, meta
    
    @classmethod
    # 这边是创建生成数据集的存储位置
    def init(cls, init_params):
        print("Dataset Init!")
        root_dir_target = init_params['root_dir']
        root_dir_target = Path(root_dir_target)
        root_dir_target.mkdir(parents=True,exist_ok=True)
        (root_dir_target).mkdir(parents=True,exist_ok=True)
        dataset_params = {
            'root_dir': str(root_dir_target),
        }
        return dataset_params
    
    @classmethod
    # TODO:这地方还没写完，因为暂时流程没有到这里，先搁置
    def add(cls, ecg_data_new, label_new, meta_new, dataset_info):
        dataset_info = dataset_info.copy() # 可以不用
        root_dir = dataset_info['root_dir']
        root_dir = Path(root_dir)
        ecg_data_path_str = meta_new['path']
        ecg_data_output_path = root_dir/ecg_data_path_str.split('/')[-2]/ecg_data_path_str.split('/')[-1]
        ecg_data_output_path.parent.mkdir(parents=True,exist_ok=True)
        label_output_path = root_dir/ecg_data_path_str.split('/')[-2]/ecg_data_path_str.split('/')[-1].replace('data','label')
        label_output_path.parent.mkdir(parents=True,exist_ok=True)
        # 保存新数据和标签
        if isinstance(ecg_data_new,torch.Tensor):
            ecg_data_new = ecg_data_new.to('cpu').numpy()
            label_new = label_new.to('cpu').numpy()
        with open(ecg_data_output_path,'w'):
            np.save(ecg_data_output_path,ecg_data_new)
        with open(label_output_path,'w'):
            np.save(label_output_path,label_new)
        return dataset_info