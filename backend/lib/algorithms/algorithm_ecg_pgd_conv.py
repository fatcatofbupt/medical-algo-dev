from cProfile import label
from unittest import result
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .factory import TestFactory

def sap_conv(
        inputs, 
        lengths, 
        targets, 
        model,
        criterion=F.cross_entropy, 
        eps = 0.03, 
        step_alpha = 0.001, 
        num_steps = 20, 
        sizes = [5, 7, 11, 15, 19],
        sigmas = [1.0, 3.0, 5.0, 7.0, 10.0],
        max_sentence_length = 187
        ):
        """
        :param inputs: Clean samples (Batch X Size)
        :param targets: True labels
        :param model: Model
        :param criterion: Loss function
        :param gamma:
        :return:
        """
        # 默认从cpu加载数据
        device = 'cpu'
        
        crafting_sizes = []
        crafting_weights = []
        for size in sizes:
            for sigma in sigmas:
                crafting_sizes.append(size)
                weight = np.arange(size) - size//2
                weight = torch.tensor(weight)
                sigma = torch.tensor(sigma)
                a = torch.exp(-weight**2.0/2.0/(sigma**2))
                weight = a/torch.sum(a)
                weight = weight.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
                crafting_weights.append(weight)
        crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
        crafting_target = torch.autograd.Variable(targets.clone())
        ##测试模型，生成模型预测真实样本结果
        true_result = F.softmax(model(crafting_input), dim=1)
        trueresult = true_result.max(1, keepdim=True)[1]
        for i in range(num_steps):
            output = model(crafting_input)
            loss = criterion(output, crafting_target)
            if crafting_input.grad is not None:
                crafting_input.grad.data.zero_()
            loss.backward()
            added = torch.sign(crafting_input.grad.data)
            step_output = crafting_input + step_alpha * added
            total_adv = step_output - inputs
            total_adv = torch.clamp(total_adv, -eps, eps)
            crafting_output = inputs + total_adv
            crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
        added = crafting_output - inputs
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        for i in tqdm(range(num_steps*2)):
            temp = F.conv1d(added, crafting_weights[0], padding = crafting_sizes[0]//2)
            for j in range(len(crafting_sizes)-1):
                temp = temp + F.conv1d(added, crafting_weights[j+1], padding = crafting_sizes[j+1]//2)
            temp = temp/float(len(crafting_sizes))
            output = model(inputs + temp)
            loss = criterion(output, targets)
            loss.backward()
            added = added + step_alpha * torch.sign(added.grad.data)
            added = torch.clamp(added, -eps, eps)
            added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        temp = F.conv1d(added, crafting_weights[0], padding = crafting_sizes[0]//2)
        for j in range(len(crafting_sizes)-1):
            temp = temp + F.conv1d(added, crafting_weights[j+1], padding = crafting_sizes[j+1]//2)
        temp = temp/float(len(crafting_sizes))
        crafting_output = inputs + temp.detach()
        crafting_output_clamp = crafting_output.clone()
        #生成模型预测对抗样本结果
        outputs = F.softmax(model(crafting_output_clamp), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        result = torch.cat((trueresult,predicted))
        return  crafting_output_clamp,result

def pgd_conv(
        inputs, 
        lengths, 
        targets, 
        model,
        criterion=F.cross_entropy, 
        eps = 0.03, 
        step_alpha = 0.001, 
        num_steps = 20, 
        max_sentence_length = 187
        ):
        """
        :param inputs: Clean samples (Batch X Size)
        :param targets: True labels
        :param model: Model
        :param criterion: Loss function
        :param gamma:
        :return:
        """
        # 默认从cpu加载数据
        device = 'cpu'
        crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
        crafting_target = torch.autograd.Variable(targets.clone())
        ##测试模型，生成模型预测真实样本结果
        true_result = F.softmax(model(crafting_input), dim=1)
        trueresult = true_result.max(1, keepdim=True)[1]
        for i in range(num_steps):
            output = model(crafting_input)
            loss = criterion(output, crafting_target)
            if crafting_input.grad is not None:
                crafting_input.grad.data.zero_()
            loss.backward()
            added = torch.sign(crafting_input.grad.data)
            step_output = crafting_input + step_alpha * added
            total_adv = step_output - inputs
            total_adv = torch.clamp(total_adv, -eps, eps)
            crafting_output = inputs + total_adv
            crafting_input = torch.autograd.Variable(crafting_output.clone(), requires_grad=True)
        added = crafting_output - inputs
        crafting_output = inputs+ added
        crafting_output_clamp = crafting_output.clone()
        outputs = F.softmax(model(crafting_output_clamp), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        result = torch.cat((trueresult,predicted))
        return  crafting_output_clamp,result

@TestFactory.register('algorithm_ecg_adv')
class Attack_ecg_pgd_conv(object):
    # TODO:待补充
    params = {

    }
    
    def __init__(self,attack_types = 'pgd_conv',num_steps = 20):
        super(Attack_ecg_pgd_conv, self).__init__()
        self.criterion = F.cross_entropy
        self.attack_types = attack_types
        self.num_steps = num_steps

    def run(self, ecg_data,label,meta,model,device = 'cpu'):
        # 攻击方式待补充
        if self.attack_types == 'sap_conv':
            adv_ecg_data,adv_ecg_label = sap_conv(
                                        inputs = ecg_data, 
                                        lengths = torch.tensor(meta['length']).unsqueeze(0), 
                                        targets = label, 
                                        model = model,
                                        criterion=self.criterion, 
                                        eps = 0.03, 
                                        step_alpha = 0.001, 
                                        num_steps = self.num_steps, 
                                        sizes = [5, 7, 11, 15, 19],
                                        sigmas = [1.0, 3.0, 5.0, 7.0, 10.0],
                                        max_sentence_length = 187
                                    )
        if self.attack_types == 'pgd_conv':
            adv_ecg_data,adv_ecg_label = pgd_conv(
                                        inputs = ecg_data, 
                                        lengths = torch.tensor(meta['length']).unsqueeze(0), 
                                        targets = label, 
                                        model = model,
                                        criterion=self.criterion, 
                                        eps = 0.03, 
                                        step_alpha = 0.001, 
                                        num_steps = self.num_steps, 
                                        max_sentence_length = 187
                                    )
        new_meta = meta.copy()
        new_meta['score'] = 0
        return adv_ecg_data,adv_ecg_label,new_meta