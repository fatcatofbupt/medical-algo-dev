import torch
import torch.nn as nn
import requests
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import json
import os

from .factory import ModelFactory

class MulticlassBinaryModel(nn.Module):
    """docstring for MulticlassBinaryModel"""
    def __init__(self, model, output_indices, model_output_type='logits'):
        super(MulticlassBinaryModel, self).__init__()
        self.model = model
        self.output_indices = output_indices
        self.model_output_type = model_output_type

    def forward(self, x):
        output = self.model(x)
        if self.model_output_type == 'logits':
            output = torch.softmax(output,dim=-1)
        output = output[...,self.output_indices].sum(dim=-1, keepdim=True)
        other = 1 - output
        output = torch.cat([other,output],dim=-1)
        if self.model_output_type == 'logits':
            output = torch.log(output)
        return output


@ModelFactory.register('model_fundus_binary')
class BinaryClassificationImagePytorchModel(object):
    params = {
        'model_path':{
            "description":"已训练的模型参数的路径，文件以.pth结尾",
            "value_type":"string",
            "value_range":"正确路径即可",
            "default":"models/jit_module_448_cpu.pth"
        },
        'model_output_type':{
            "description":"模型输出数据类型",
            "value_type":"enum",
            "value_range":['logits','ligit','probabilities','probability'],
            "default":"logits"
        },
        'threshold':{
            "description":"模型输出为正例的阈值",
            'value_type':"float",
            "value_range":"0~1之间的浮点数",
            "default":0.5
        },
        'device':{
            "description":"算法运行的设备",
            "value_type":"enum",
            "value_range":["cpu","cuda"],
            "default":"cpu"
        },
        'multi2binary_indices':{
            "description":"将多分类模型输出变成2分类，若模型已是2分类，则输入None",
            "value_type":"list",
            "value_range":"输入为正类别的序号，如一共有5个类别，则序号为不超过4的值，通常取为[1,2,3,4](也可以取为其他值，如[2,3,4]),如果模型已经是二分类模型，则取为None",
            "default": [1,2,3,4]
        }
    }
    def __init__(self, path ,output_type='logits', 
                       threshold=0.5, device='cpu',
                       multi2binary_indices=None,
                       ):
        '''
        output_type: str
            说明模型输出数据类型
            - logits: array[float] (添加softmax后处理)
            - logit: float (添加sigmoid后处理)
            - probabilities: array[float]
            - probability: float
        '''
        super(BinaryClassificationImagePytorchModel, self).__init__()
        from torchvision.transforms import transforms

        # token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyNDVlODczYS1kMDRhLTQzODQtOWY0OS0wZDIzZDdkZGI0MGYiLCJpYXQiOjE2MzQ1NDkyNzh9.JwQ3aoMnBjAU92GZeedW1S8f7eDpRhTfNwJYahgb94Y'
        # url = "http://81.70.116.29/rest/cos/federationToken/"
        # headers = {
        #     'Content-Type': 'application/json;charset=UTF-8',
        #     'Authorization': token
        # }
        # response = requests.get(url,headers=headers)
        # response = json.loads(response.text)

        # secret_id = response['Credentials']['TmpSecretId']
        # secret_key = response['Credentials']['TmpSecretKey']
        # region = 'ap-beijing'
        # self.bucket = 'adversial-attack-1307678320'
        # token = response['Credentials']['Token']
        # scheme = 'https'
        # config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
        
        # self.client = CosS3Client(config)
        # resp = self.client.get_object(Bucket=self.bucket,Key=path)
        # print("RESP:",resp)
        # fp = resp['Body'].get_raw_stream()
        # file_dir = os.path.join(download_dir,'model.pth')
        # with open(file_dir,'wb') as f:
        #     f.write(fp.read())
        self.model_output_type = output_type
        self.threshold = threshold
        self.model = torch.jit.load(path).to(device)
        # self.model = torch.jit.load(str(path))
        self.device = device
        self.multi2binary_indices = multi2binary_indices
        if not (self.multi2binary_indices is None):
            self.model = MulticlassBinaryModel(self.model, 
                output_indices=multi2binary_indices, 
                model_output_type=output_type)

        self._to_tensor = transforms.ToTensor()

    def predict(self, x):
        '''
        x: np.array[H,W,3]
        '''
        x = self._to_tensor(x)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        output = self.model(x)
        if self.model_output_type == 'logits':
            output = torch.softmax(output,dim=-1)
            output = output[...,1]
        elif self.model_output_type == 'logit':
            output = torch.sigmoid(output)
            if len(output.shape) == 2:
                output = output[...,0]
        elif self.model_output_type == 'probabilities':
            output = output[...,1]
        elif self.model_output_type == 'probability':
            if len(output.shape) == 2:
                output = output[...,0]
        assert len(output.shape) == 1
        prob = output.item()
        pred = int(prob >= self.threshold)
        return pred, prob
