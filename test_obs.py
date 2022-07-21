from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import logging
import requests 
import json

from torch.utils.data import dataset

token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyNDVlODczYS1kMDRhLTQzODQtOWY0OS0wZDIzZDdkZGI0MGYiLCJpYXQiOjE2MzQ1NDkyNzh9.JwQ3aoMnBjAU92GZeedW1S8f7eDpRhTfNwJYahgb94Y'
url = "http://81.70.116.29/rest/cos/federationToken/"
headers = {
    'Content-Type': 'application/json;charset=UTF-8',
    'Authorization': token
}
response = requests.get(url,headers=headers)
# print(response.text)
response = json.loads(response.text)

secret_id = response['Credentials']['TmpSecretId']
secret_key = response['Credentials']['TmpSecretKey']
print(secret_id)
print(secret_key)
region = 'ap-beijing'
bucket = 'adversial-attack-1307678320'
token = response['Credentials']['Token']
scheme = 'https'
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
client = CosS3Client(config)
response = client.list_objects(Bucket=bucket, Prefix='dataset/')
# response = client.list_objects(Bucket=bucket,Prefix='generated_dataset/')
print(response)
# response = client.get_object(
#     Bucket=bucket,
#     Key='dataset/fundus/images/35598_right.jpeg', # 可以包含文件路径
# )
# response['Body'].get_stream_to_file('tmp2.jpg')

# response = client.get_object(
#     Bucket=bucket,
#     Key='dataset/fundus/images/35598_right.jpeg',
# )
# fp = response['Body'].get_raw_stream()
# with open('tmp.jpg','wb') as f:
#     f.write(fp.read())
# print(fp.read(2))

# print("Start Uploading...")
# with open('test/tmp_zip/16351748997211578.zip','rb') as f:
#     response = client.put_object(
#         Bucket=bucket,
#         Body = f,
#         Key='generated_dataset/' + 'test.zip', # 上传后的文件名称，可以包含文件路径
#         StorageClass='Standard',
#     )
# print(response['ETag'])
# print("Upload Finished!")
# print(response)