import torch
from torch._C import device
import torch.nn as nn
import requests
from torchvision.transforms import transforms
import copy

from backend.lib.models.factory import ModelFactory

@ModelFactory.register('model_lungseg_basic')
class LungSegBasic(nn.Module):
    params = {
        'model_path':{
            "description":"已训练的模型参数的路径，文件以.pth结尾",
            "value_type":"string",
            "value_range":"正确路径即可",
            "default":"models/jit_module_448_cpu.pth"
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
        }
    }
    def __init__(self,path,threshold=0.5, device='cpu'):
        super(LungSegBasic, self).__init__()
        self.model = torch.jit.load(path,map_location=device).to(device)
        self.threshold = threshold
        self.device = device
        self._to_tensor = transforms.ToTensor()

    def forward(self, x):
        '''
        x: np.array[H,W,3]
        '''
        # x = self._to_tensor(x)
        # print("Input shape:", x.shape)
        # print("Model Device:", self.device)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # print(self.model)
        # print(x)
        output = self.model(x)
        prob = torch.sigmoid(output)
        # pred = torch.tensor(output)
        pred = torch.zeros(output.shape)
        # pred = copy.deepcopy(output)
        pred[prob >= self.threshold] = 1
        pred[prob < self.threshold] = 0
        # print("Output shape:", pred.shape)
        
        return pred, prob

    def predict(self, x):
        if len(x.shape) == 3:
            x = self._to_tensor(x)
            x = x.unsqueeze(0).to(self.device)
        output = self.model(x)
        prob = torch.sigmoid(output)
        pred = torch.zeros(output.shape)
        # pred = copy.deepcopy(output)
        pred[prob >= self.threshold] = 1
        pred[prob < self.threshold] = 0
        return pred.cpu().detach().numpy(),prob.cpu().detach().numpy()
