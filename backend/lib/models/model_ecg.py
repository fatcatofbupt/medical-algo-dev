import torch
from torch._C import device
import torch.nn as nn
import requests
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F

from .factory import ModelFactory


@ModelFactory.register('model_ecg_classifiction')
class ECG_Classion(nn.Module):
    # TODO:待补充
    params = {
    }
    def __init__(self,path, device='cpu',model_output_type = 'logits'):
        super(ECG_Classion, self).__init__()
        #加载模型
        self.model = torch.jit.load(path,map_location=device).to(device)

        # 关闭模型的drop和batchnorm
        self.model.eval()

        # 设备默认cpu
        self.device = device

        # softmax
        self.softmax = F.softmax

        self.model_output_type = model_output_type
    def forward(self, x):
        output = self.model(x)
        return output

    def predict(self,x):
        output = self.forward(x)
        if self.model_output_type == 'logits':
            output = torch.softmax(output,dim=-1)
            prob = output.max()
        # TODO:目前只看了logits
        # elif self.model_output_type == 'logit':
        #     output = torch.sigmoid(output)
        #     if len(output.shape) == 2:
        #         output = output[...,0]
        # elif self.model_output_type == 'probabilities':
        #     output = output[...,1]
        # elif self.model_output_type == 'probability':
        #     if len(output.shape) == 2:
        #         output = output[...,0]
        pred = output.argmax().item()
        return pred, prob