from obs import ObsClient
import os
import cv2
obsClient = ObsClient(
        access_key_id='0YGLVBGF0NWSRF55MSX5',    
        secret_access_key='jZ664BogYajhpDBgeMJX9aTJCrIJLdWPOb3EbkjI',    
        server='https://obs.cn-north-4.myhuaweicloud.com'
    )
img_dir = './test/fundus/images'
label_dir = './test/fundus'
obs_dir = 'topic4/fundus/images'
obs_ldir = 'topic4/fundus'
images = os.listdir(img_dir)
# for image in images:
#     obs_name = os.path.join(obs_dir,image)
#     local_name = os.path.join(img_dir,image)
#     resp1 = obsClient.uploadFile('zhongjianyuan',obs_name,local_name)
obs_lname = os.path.join(obs_ldir,'labels.json')
local_lname = os.path.join(label_dir,'labels.json')
respl = obsClient.uploadFile('zhongjianyuan',obs_lname,local_lname)
resp2 = obsClient.listObjects('zhongjianyuan',marker=obs_ldir,max_keys=12)
print(resp2)
# print(resp1)