import numpy as np
import random
import torch
from .factory import TestFactory
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms
# cyelegan预处理
img_to_tensor = transforms.ToTensor()
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



@TestFactory.register('general_lungtog')
class GeneralLungssd(object):
    params = {
        'attack_types':{
            "description":"选择攻击的类型,从value_range中选择任意一数的攻击类型输入",
            "value_type":"list",
            "value_range":['Untargeted_Random','Object-vanishing','Object-fabrication','Object-mislabeling'],
            "default": 'Object-vanishing'
        },
        'attack_levels':{
            "description":"加的攻击扰动范围",
            "value_type":"int",
            "value_range":"0-255之间的float",
            "default":8
        },
        'n_iter':{
            "description":"攻击次数",
            "value_type":"int",
            "value-range":"合理即可，越大攻击花费时间越久，效果越好，一般20即可",
            "default":20
        }
    }
    def __init__(self, attack_types='Object-vanishing', attack_levels=8,n_iter = 20,eps_iter = 0.1):
        super(GeneralLungssd, self).__init__()
        self.attack_types = attack_types
        self.attack_levels = attack_levels
        self.n_iter = n_iter
        self.eps_iter = eps_iter
    def tog_vanishing(self,
                        victim,
                        x_query
                        ):
        eps = self.attack_levels
        n_iter = self.n_iter
        eps_iter = self.eps_iter
        eta = eps*(torch.rand_like(x_query)-0.5)
        x_adv = np.clip(x_query + eta, -123.0, 151.0)
        for _ in range(n_iter):
            grad = victim.compute_object_vanishing_gradient(x_adv)
            signed_grad = torch.sign(grad)
            x_adv -= eps_iter * signed_grad
            eta = torch.clip(x_adv - x_query, -eps, eps)
            x_adv = torch.clip(x_query + eta, -123.0, 151.0)
        return x_adv
    def single_attack(self,img,model,device):
        # 这边考虑到医院肺结节一般都是长宽相等的，resize时候只考虑了正方形
        w = img.shape[0]
        # 一些预处理和反预处理的类方法继承
        # resize 尺寸到300，并且归一化
        transform = BaseTransform(model.size, (104, 117, 123))
        # resize 尺寸到原尺寸，并且反归一化
        untransform = BaseTransform(w,(-104,-117,-123))
        # 图片预处理以及一些数据的处理
        pre_treated_image = torch.from_numpy(transform(img)).permute(2, 0, 1).unsqueeze(0)
       
        # Move to default device
        pre_treated_image = pre_treated_image.to(device)

        x_adv = self.tog_vanishing(model,pre_treated_image)
        # 将加的扰动后的图片变成原图规模
        img_adv = untransform(x_adv.squeeze(0).permute(1,2,0).numpy())
        # 将原图变成原图规模
        img_ori = untransform(pre_treated_image.squeeze(0).permute(1,2,0).numpy())
        # 去模糊原图，加清晰原图
        img_adv = img_adv - img_ori + img
        # 调整照片np数组数据类型以及去掉超过阈值的部分，就变成了被攻击之后的最终的照片
        img_adv = np.clip(img_adv,0,255).astype(np.uint8)
        return img_adv

    def run(self, img, label, meta, model, device):
        # 目标检测结果
        img_adv = img
        for i in range(10): 
            det = model.detect_a_img(img_adv)
            if(det['scores'] == []):
                break
            img_adv = self.single_attack(img_adv,model,device)

        # detect_results包含了攻击之后的检测结果和原始的标注信息
        detect_results = {'attacked_boxes':det['coords'],
                 'attacked_scores':det['scores'],
                 'attacked_labels':det['labels'],
                 'true_boxes':label['boxes'],
                 'true_labels':label['labels'],
                 'true_difficults':label['difficults']
                 }
        # TODO:这边new_meta先等于这个，明天看看具体该放什么
        new_meta = meta
        return img_adv, detect_results, new_meta

@TestFactory.register('algorithm_lungtog_cyclegan')
class LungssdCycleGAN(nn.Module):
    #TODO 待填充
    params = {
        'a_path':{
            "description":"从无病生成有病图像的GAN的路径",
            "value_type":"str",
            "value_range":"无",
            "default": './models/net_G_A.pth'
        },
        'b_path':{
            "description":"从有病生成无病图像的GAN的路径",
            "value_type":"str",
            "value_range":"无",
            "default": './models/net_G_B.pth'
        }
    }
    def __init__(self,a_path,b_path,device='cpu'):
        super(LungssdCycleGAN,self).__init__()
        print("Loading Jit Model...")
        self.generatorAtoB = torch.load(a_path)
        # print(self.generatorAtoB)
        self.generatorBtoA = torch.load(b_path)
        self.device = device
    
    def run(self, img, label, meta, model, device):
        try:
            model = model.to(device)
        except Exception as e:
            model = model.model
            model = model.to(device)
        img = img_to_tensor(img)
        img = img.unsqueeze(0).to(device)
        preds_ori = model(img)
        img = img.squeeze(0)
        transform_list = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        preprocess = transforms.Compose(transform_list)
        # print('imshape:',img.shape)
        # img = pad_img(img)
        img = preprocess(img)
        img = img.unsqueeze(0).to(device)


        if label == 0:
            fake_img = self.generatorAtoB(img.to(device))
            fake_label = 1
        else:
            fake_img = self.generatorBtoA(img.to(device))
            fake_label = 0
        # print(fake_img)

        new_meta = meta.copy()
        preds_new = model(fake_img)
        preds_ori = torch.argmax(preds_ori,dim=1).item()
        preds_new = torch.argmax(preds_new,dim=1).item()
        new_meta['label'] = 1 if label >= 1 else 0
        new_meta['new_label'] = fake_label
        new_meta['pred_ori'] = preds_ori
        new_meta['pred_new'] = preds_new

        fake_img = fake_img.squeeze(0)
        image_numpy = fake_img.detach().float().cpu().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
        image_pil = Image.fromarray(np.uint8(image_numpy))
        image_pil.save('fake.jpg')
        return np.uint8(image_numpy),fake_label,new_meta
