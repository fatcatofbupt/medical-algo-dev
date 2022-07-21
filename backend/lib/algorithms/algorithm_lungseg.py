import albumentations
import torchvision.transforms as transforms
import numpy as np
import random
import torch

from .factory import TestFactory


img_to_tensor = transforms.ToTensor()



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


@TestFactory.register('general_lungseg')
class GeneralLungseg(object):
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
        super(GeneralLungseg, self).__init__()
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
        new_meta = meta.copy()
        img = img_to_tensor(img)
        # print("Algorithm Device:",device)
        img = img.unsqueeze(0).to(device)
        y = torch.tensor([label]).to(device)
        with torch.no_grad():
            preds_ori = model.predict(img)
        adv_tensor = img_to_tensor(adv_img)
        adv_tensor = adv_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            preds_new = model.predict(adv_tensor)
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        # torch.cuda.empty_cache()
        new_meta['pred_ori'] = preds_ori
        new_meta['pred_new'] = preds_new
        new_meta['label'] = label
        return adv_img, label, new_meta
