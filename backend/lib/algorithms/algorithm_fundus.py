import albumentations
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torch.jit
import torch.nn as nn

from .factory import TestFactory
from .fundus_pytorch import PGD
import cv2
from PIL import Image

img_to_tensor = transforms.ToTensor()


#Add gaussian noise to an image
def add_gaussian_noise(pic,noise_sigma=[10,50],p=0.3):
    num = np.random.rand()
    if num >= p:
        return pic
    temp_image = np.float64(np.copy(pic))
    sigma_range = np.arange(noise_sigma[0],noise_sigma[1])
    sigma = random.sample(list(sigma_range),1)[0]
    h, w, _ = temp_image.shape
    #Standard norm * noise_sigma
    noise = np.random.randn(h, w) * sigma
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
        
    return noisy_image

#Add salt and pepper noise to an image
def add_sp_noise(pic,SNR=0.9,p=0.3):
    num = np.random.rand()
    if num >= p:
        return pic
    noisy_image = pic.copy()
    h, w, c = noisy_image.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)     # copy the mask by channel
    mask = np.transpose(mask,(1,2,0))
    noisy_image[mask == 1] = 255    # Salt
    noisy_image[mask == 2] = 0      # Pepper
    return noisy_image


@TestFactory.register('algorithm_fundus_general')
class FundusImageSimPerturb(object):
    params = {
        'attack_types':{
            "description":"选择扰动的类型,从value_range中选择任意个数的扰动类型输入",
            "value_type":"list",
            "value_range":['defocus_blur','motion_blur','rgb_shift','rgb_shift','hsv_shift',"brightness_shift","iso_noise","sp_noise"],
            "default": ['defocus_blur','iso_noise']
        },
        'attack_levels':{
            "description":"每个扰动的等级，需要与上述扰动一一对应",
            "value_type":"list",
            "value_range":"1-5之间的整数",
            "default":[2,3]
        },
    }
    def __init__(self, attack_types=['defocus_blur','iso_noise'], attack_levels=[2,3]):
        super(FundusImageSimPerturb, self).__init__()
        self.attack_types = attack_types
        self.attack_levels = attack_levels
        # self.params = 0
    # @classmethod
    # def params(self):
    #     return 0
    def run(self, img, label, meta, model, device):
        attack_dict = {}
        attack_list = []
        assert(len(self.attack_types)==len(self.attack_levels))
        for i,type in enumerate(self.attack_types):
            attack_dict[type] = self.attack_levels[i]
        if 'defocus_blur' in self.attack_types:
            level_list = [1,3,5,7,9]
            level = attack_dict['defocus_blur']
            blur_limit = level_list[int(level)-1]
            attack_list.append(albumentations.GaussianBlur(blur_limit=blur_limit, p=1))
        if 'motion_blur' in self.attack_types:
            level_list = [10,30,50,70,90]
            level = attack_dict['motion_blur']
            blur_limit = level_list[int(level)-1]
            attack_list.append(albumentations.MotionBlur(blur_limit=blur_limit,p=1))
        if 'rgb_shift' in self.attack_types:
            attack_list.append(albumentations.RGBShift(p=1))
        if 'hsv_shift' in self.attack_types:
            level_list = [5,10,15,20,25]
            level = attack_dict['hsv_shift']
            shift_limit = level_list[int(level)-1]
            attack_list.append(albumentations.HueSaturationValue(hue_shift_limit=shift_limit, sat_shift_limit=shift_limit, val_shift_limit=shift_limit, p=1))
        if 'brightness_contrast' in self.attack_types:
            level_list = [0.1,0.2,0.3,0.4,0.5]
            level = attack_dict['brightness_contrast']
            limit = level_list[int(level)-1]
            attack_list.append(albumentations.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=1))
        album = albumentations.Compose(attack_list)

        transform = transforms.ToTensor()
        reverse = transforms.ToPILImage()
        # img = item['image']
        # label = item['label']
        # img = reverse(img.squeeze())
        # img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        adv_img = album(image=img)["image"]
        if 'iso_noise' in self.attack_types:
            mean_list = [2,5,10,15,20]
            sigma_list = [30,40,50,60,70]
            level = attack_dict['iso_noise']
            idx = int(level) - 1
            adv_img = add_gaussian_noise(adv_img,[mean_list[idx],sigma_list[idx]],p=1)
        if 'sp_noise' in self.attack_types:
            level_list = [0.9,0.8,0.7,0.6,0.5]
            level = attack_dict['sp_noise']
            snr = level_list[int(level)-1]
            adv_img = add_sp_noise(adv_img,SNR=snr,p=1)
        # adv_img = cv2.cvtColor(adv_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # adv_img = transform(adv_img.astype(np.uint8))

        try:
            model = model.to(device)
        except Exception as e:
            model = model.model
            model = model.to(device)

        new_meta = meta.copy()
        img = img_to_tensor(img)
        img = img.unsqueeze(0).to(device)
        y = torch.tensor([label]).to(device)
        preds_ori = model(img)
        adv_tensor = img_to_tensor(adv_img)
        adv_tensor = adv_tensor.unsqueeze(0).to(device)
        preds_new = model(adv_tensor)
        preds_ori = torch.argmax(preds_ori,dim=1).item()
        preds_new = torch.argmax(preds_new,dim=1).item()
        new_meta['pred_ori'] = preds_ori
        new_meta['pred_new'] = preds_new
        new_meta['label'] = 1 if label >= 1 else 0
        return adv_img, label, new_meta


@TestFactory.register('algorithm_fundus_adv_pgd_pytorch')
class FundusImagePGDAdversial(object):
    params = {
        'adv_level':{
            "description":"选择对抗攻击的等级",
            "value_type":"int",
            "value_range":"1-3之间的整数或-1，-1表示测试",
            "default": -1
        },
    }
    def __init__(self, adv_level):
        super(FundusImagePGDAdversial, self).__init__()
        self.level = adv_level
        self.set_attack_level()

    # @classmethod
    # def params(self):
    #     return 0

    def set_attack_level(self):
        if self.level == -1:
            self.eps=0.005
            self.iter_eps=0.001
            self.nb_iter=1 # for debug
        elif self.level == 1:
            # print('Level:1')
            self.eps=0.005
            self.iter_eps=0.001
            self.nb_iter=20
        elif self.level == 2:
            # print('Level:2')
            self.eps=0.02
            self.iter_eps = 0.002
            self.nb_iter = 40
        elif self.level == 3:
            # print('Level:3')
            self.eps = 0.1
            self.iter_eps = 0.003
            self.nb_iter = 60

    def run(self, img, label, meta, model, device):
        try:
            model = model.to(device)
        except Exception as e:
            model = model.model
            model = model.to(device)
        attacker = PGD(model,device) 
        # h, w, _ = img.shape
        img = img_to_tensor(img)
        img = img.unsqueeze(0).to(device)
        y = torch.tensor([label]).to(device)
        preds_ori = model(img)
        adv_img = attacker.generate(img,y=y,eps=self.eps,iter_eps=self.iter_eps,nb_iter=self.nb_iter,mask=None)
        preds_new = model(adv_img)
        adv_img = adv_img.squeeze(0)
        new_meta = meta.copy()
        # print("Original Prediction:",preds_ori)
        # print("New Prediction:",preds_new)
        preds_ori = torch.argmax(preds_ori,dim=1).item()
        preds_new = torch.argmax(preds_new,dim=1).item()
        # print("Original Pred Label:",preds_ori)
        # print("New Pred Label:",preds_new)
        new_meta['pred_ori'] = preds_ori
        new_meta['pred_new'] = preds_new
        # binary_label = copy.deepcopy(label)
        # binary_label[binary_label>1] = 1
        new_meta['label'] = 1 if label >= 1 else 0
        adv_img = np.array(torchvision.transforms.ToPILImage()(adv_img))
        return adv_img, label, new_meta

@TestFactory.register('algorithm_fundus_cyclegan')
class FundusCycleGAN(nn.Module):
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
        super(FundusCycleGAN,self).__init__()
        print("Loading Jit Model...")
        self.generatorAtoB = torch.jit.load(a_path).to(device)
        # print(self.generatorAtoB)
        self.generatorBtoA = torch.jit.load(b_path).to(device)
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
        


