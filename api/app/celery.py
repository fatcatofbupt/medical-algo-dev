import sys
sys.path.append('./')
from celery import Celery
from flask import current_app
from backend.lib.core import BenchmarkerCore
celery_app = Celery(__name__)

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import time
import copy
import os
import requests
import json


@celery_app.task(bind=True)
def test_all(self,params):
    try:
        algorithm = params.pop('algorithm')
        tests = algorithm['tests']
    except:
        # print("Execption!")
        return {"Error":"Parames not correct(algorithm params not right)"}
    task_ids = []
    for test in tests:
        params['algorithm'] = {}
        params['algorithm']['test'] = test
        task = test_and_create.apply_async(args=[params])
        task_ids.append(task.id)
        # print(task.id)
    return [id for id in task_ids]

@celery_app.task(bind=True)
def test_and_create(self,params):
    benchmarker = BenchmarkerCore()
    try:
        new_params = make_new_params(params,evaluate_only=False)
    except:
        return {"Error":"Parames not correct"}
    result = benchmarker.create_dataset(new_params,use_obs=True)
    self.update_state(state='SUCCESS',meta=result)
    return result

@celery_app.task(bind=True)
def evaluate(self,params):
    benchmarker = BenchmarkerCore()
    try:
        new_params = make_new_params(params,evaluate_only=True)
    except:
        return {"Error":"Parames not correct"}
    result = benchmarker.evaluate_only(new_params,use_obs=True)
    self.update_state(state='SUCCESS',meta=result)
    return result


def make_new_params(params,evaluate_only = False):
    params = copy.deepcopy(params)
    now_time = str(time.time()).replace('.','')
    tmp_dir = 'test/tmp/' + now_time
    model_tmp_dir = 'test/tmp/' + now_time + '/model/'
    os.makedirs(tmp_dir,exist_ok=True)
    os.makedirs(model_tmp_dir,exist_ok=True)
    dataset_info = params['dataset']
    dataset_name = dataset_info['name']
    dataset_params = dataset_info['params']
    model_info = params['model']
    model_name = model_info['name']
    model_params = model_info['params']
    
    params['dataset_name'] = dataset_name
    params['dataset_params'] = dataset_params
    params['model_name'] = model_name
    params['model_params'] = model_params
    params['tmp_dir'] = tmp_dir
    params['model_tmp_dir'] = model_tmp_dir
    bucket,client = obs_init()
    params['bucket'] = bucket
    params['client'] = client

    if not evaluate_only:
        out_dir = 'test/tmp_out/' + now_time
        zip_dir = 'test/tmp_zip/'
        zip_file = zip_dir + now_time + '.zip'
        test_info = params['algorithm']
        test_name = test_info['test']['name']
        test_params = test_info['test']['params']
        params['test_name'] = test_name
        params['test_params'] = test_params
        params['out_dir'] = out_dir
        params['zip_file'] = zip_file
        os.makedirs(out_dir,exist_ok=True)
        os.makedirs(zip_dir,exist_ok=True)
        param_string = ''
        for param in test_params:
            content = test_params[param]
            param_string += param + str(content) + "_"
        upload_name = dataset_name + "_" + model_name + "_" + test_name + "_" + param_string + now_time + '.zip'
        params['upload_name'] = upload_name 
    else:
        script_tmp_dir = 'test/tmp/' + now_time + '/script.py'
        params['script_tmp_dir'] = script_tmp_dir
    return params

def obs_init():
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyNDVlODczYS1kMDRhLTQzODQtOWY0OS0wZDIzZDdkZGI0MGYiLCJpYXQiOjE2MzQ1NDkyNzh9.JwQ3aoMnBjAU92GZeedW1S8f7eDpRhTfNwJYahgb94Y'
    url = "http://81.70.116.29/rest/cos/federationToken/"
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'Authorization': token
    }
    response = requests.get(url,headers=headers)
    response = json.loads(response.text)

    secret_id = response['Credentials']['TmpSecretId']
    secret_key = response['Credentials']['TmpSecretKey']
    region = 'ap-beijing'
    bucket = 'adversial-attack-1307678320'
    token = response['Credentials']['Token']
    scheme = 'https'
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
    client = CosS3Client(config)

    return bucket, client

# @celery_app.task(bind=True)
# def adv_attack(self,upload_files,level,labels):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     imgs = []
#     # for img_f in upload_files:
#     #     img_f = img_f.read()
#     #     im = cv2.imdecode(np.frombuffer(img_f, np.uint8), cv2.IMREAD_COLOR)
#     #     imgs.append(im)
#     fail_to_read = []
#     fail_to_attack = []
#     for img_f in upload_files:
#         resp = obsClient.getObject('zhongjianyuan',img_f,'./test/{}'.format(img_f.split('/')[-1]))
#         if resp.status < 300:
#             im = cv2.imread('./test/{}'.format(img_f.split('/')[-1]))
#             imgs.append(im)
#             print('Read Success!')
#         else:
#             fail_to_read.append(img_f)
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     assert(len(labels)==len(imgs))
#     adv_imgs = []
#     if device == 'cpu':
#         model = torch.jit.load('./app/weights/jit_module_448_cpu.pth')
#     else:
#         model = torch.jit.load('./app/weights/jit_module_448_gpu.pth')
#     model = model.to(device)
#     attacker = Mask_PGD(model,device)
#     # attacker = Smooth_PGD(model,device)
#     if level == 1:
#         print('Level:1')
#         eps=0.005
#         iter_eps=0.001
#         nb_iter=20
#     elif level == 2:
#         print('Level:2')
#         eps=0.02
#         iter_eps = 0.002
#         nb_iter = 40
#     elif level == 3:
#         print('Level:3')
#         eps = 0.1
#         iter_eps = 0.003
#         nb_iter = 60
#     for count,img in enumerate(imgs):
#         img = img.astype(np.uint8)
#         img = pad_img(img)
#         h,w,c = img.shape
#         img = cv2.resize(img,(448,448))
#         transform = transforms.ToTensor()
#         reverse = transforms.ToPILImage()
#         img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
#         img = transform(img.astype(np.uint8))
#         img = img.unsqueeze(0)
#         # print(img.shape)
#         print("Start Attacking...")
#         adv_img = attacker.generate(img.to(device),y=torch.tensor([labels[count]]).to(device),eps=eps,iter_eps=iter_eps,nb_iter=nb_iter,mask=None)
#         # adv_img = attacker.generate(img.to(device),y=torch.tensor([labels[count]]).to(device),eps=eps,iter_eps=iter_eps,nb_iter=nb_iter,mask=None,sizes=sizes,sigmas=sigmas)
#         adv_img = torch.tensor(adv_img).to(device)
#         #Save adversarial image
#         img = reverse(adv_img.squeeze())
#         img = img.resize((h,w))
#         adv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

#         timestamp = int(round(time.time() * 1000))
#         outname = upload_files[count].split('/')[-1][:-4] + '_' + str(timestamp) + '.jpg'
#         cv2.imwrite('./test/{}'.format(outname),adv_img)
#         resp = obsClient.putFile('zhongjianyuan','topic4/{}'.format(outname),'./test/{}'.format(outname))
#         if resp.status < 300:
#             print('Upload Success!')
#             message = "Successfully uploaded " + outname
#             self.update_state(state='PROGRESS',
#                               meta={'current':count+1,'success':count+1 -len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'status':message})
#         else:
#             fail_to_attack.append(upload_files[count])
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     # cmd = 'rm -rf ./test'
#     # os.system(cmd)
#     return {'code':200,'current':len(upload_files),'success':len(upload_files) - len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'fail_to_read':fail_to_read,'fail_to_attack':fail_to_attack,'status':'Task completed!'}                

# @celery_app.task(bind=True)
# def norm_attack(self,upload_files,attack_types,attack_levels):
#     imgs = []
#     fail_to_read = []
#     fail_to_attack = []
#     for img_f in upload_files:
#         resp = obsClient.getObject('zhongjianyuan',img_f,'./test/{}'.format(img_f.split('/')[-1]))
#         if resp.status < 300:
#             im = cv2.imread('./test/{}'.format(img_f.split('/')[-1]))
#             imgs.append(im)
#             print('Read Success!')
#         else:
#             fail_to_read.append(img_f)
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     attack_dict = {}
#     attack_list = []
#     assert(len(attack_types)==len(attack_levels))
#     for i,type in enumerate(attack_types):
#         attack_dict[type] = attack_levels[i]
#     if 'defocus_blur' in attack_types:
#         level_list = [1,3,5,7,9]
#         level = attack_dict['defocus_blur']
#         blur_limit = level_list[int(level)-1]
#         attack_list.append(albumentations.GaussianBlur(blur_limit=blur_limit, p=1))
#     if 'motion_blur' in attack_types:
#         level_list = [10,30,50,70,90]
#         level = attack_dict['motion_blur']
#         blur_limit = level_list[int(level)-1]
#         attack_list.append(albumentations.MotionBlur(blur_limit=blur_limit,p=1))
#     if 'rgb_shift' in attack_types:
#         attack_list.append(albumentations.RGBShift(p=1))
#     if 'hsv_shift' in attack_types:
#         level_list = [5,10,15,20,25]
#         level = attack_dict['hsv_shift']
#         shift_limit = level_list[int(level)-1]
#         attack_list.append(albumentations.HueSaturationValue(hue_shift_limit=shift_limit, sat_shift_limit=shift_limit, val_shift_limit=shift_limit, p=1))
#     if 'brightness_contrast' in attack_types:
#         level_list = [0.1,0.2,0.3,0.4,0.5]
#         level = attack_dict['brightness_contrast']
#         limit = level_list[int(level)-1]
#         attack_list.append(albumentations.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=1))
#     album = albumentations.Compose(attack_list)
#     for rank,img in enumerate(imgs):
#         adv_img = album(image=img)["image"]
#         if 'iso_noise' in attack_types:
#             mean_list = [2,5,10,15,20]
#             sigma_list = [30,40,50,60,70]
#             level = attack_dict['iso_noise']
#             idx = int(level) - 1
#             adv_img = add_gaussian_noise(adv_img,[mean_list[idx],sigma_list[idx]],p=1)
#         if 'sp_noise' in attack_types:
#             level_list = [0.9,0.8,0.7,0.6,0.5]
#             level = attack_dict['sp_noise']
#             snr = level_list[int(level)-1]
#             adv_img = add_sp_noise(adv_img,SNR=snr,p=1)

#         timestamp = int(round(time.time() * 1000))
#         outname = upload_files[rank].split('/')[-1][:-4] + '_' + str(timestamp) + '.jpg'
#         cv2.imwrite('./test/{}'.format(outname),adv_img)
#         resp = obsClient.putFile('zhongjianyuan','topic4/{}'.format(outname),'./test/{}'.format(outname))
#         if resp.status < 300:
#             print('Upload Success!')
#             message = "Successfully uploaded " + outname
#             self.update_state(state='PROGRESS',
#                               meta={'current':rank+1,'success':rank+1 -len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'status':message})
#         else:
#             fail_to_attack.append(upload_files[rank])
#             print('errorCode:', resp.errorCode)
#             print('errorMessage:', resp.errorMessage)
#     return {'code':200,'current':len(upload_files),'success':len(upload_files) - len(fail_to_attack) - len(fail_to_read),'total':len(upload_files),'fail_to_read':fail_to_read,'fail_to_attack':fail_to_attack,'status':'Task completed!'} 
