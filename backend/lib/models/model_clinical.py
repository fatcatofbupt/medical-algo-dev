import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from torch.autograd import Variable
# from torch._C import device
# import torch.nn as nn
# import requests
# from torchvision.transforms import transforms
# import copy

from backend.lib.models.factory import ModelFactory


# def multi_label_accuracy(outputs, label, config, result=None):
#     if len(label[0]) != len(outputs[0]):
#         raise ValueError('Input dimensions of labels and outputs must match.')

#     outputs = outputs.data
#     labels = label.data

#     if result is None:
#         result = []
#         # result: list(Dict[str, Any]) = []

#     total = 0
#     nr_classes = outputs.size(1)

#     while len(result) < nr_classes:
#         # one: torch.tensor(Dict[str, Any]) = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
#         # input = [result, one] 
#         # result = torch.cat(input, dim=0)

#         result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

#         # one: torch.tensor(Dict[str, Any]) = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
#         # result += one

#     for i in range(nr_classes):
#         outputs1 = (outputs[:, i] >= 0.5).long()
#         labels1 = (labels[:, i].float() >= 0.5).long()
#         total += int((labels1 * outputs1).sum())
#         total += int(((1 - labels1) * (1 - outputs1)).sum())

#         if result is None:
#             continue

#         # if len(result) < i:
#         #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

#         result[i]["TP"] += int((labels1 * outputs1).sum())
#         result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
#         result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
#         result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

#     return result

# def cal_acc(logits,label):
#     if len(label.shape) == 1:
#         label = label.unsqueeze(0)
#     acc_result = multi_label_accuracy(outputs = logits, label = label, config = None ,result = None)
#     return acc_result

@ModelFactory.register('model_clinical_basic')
class ClinicalLstmBasic(nn.Module):
    params = {
        'model_path':{
            "description":"已训练的模型参数的路径，文件以.pth结尾",
            "value_type":"string",
            "value_range":"正确路径即可",
            "default":"models/jit_module_448_cpu.pth"
        },
        'device':{
            "description":"算法运行的设备",
            "value_type":"enum",
            "value_range":["cpu","cuda"],
            "default":"cpu"
        }
    }
    def __init__(self, path, device='cpu'):
        super(ClinicalLstmBasic, self).__init__()

        # self._to_tensor = transforms.ToTensor()
        # 加载模型
        self.model = torch.jit.load(path,map_location=device).to(device)
        # 关闭模型的drop和batchnorm
        self.model.train()
        # 设备默认cpu
        self.device = device



    def predict(self, data):
        result = self.model(data)

        return result
    
