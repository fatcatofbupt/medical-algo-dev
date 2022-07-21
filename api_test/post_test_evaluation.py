import requests
import json
import yaml
import base64
# Use this api to require local test service
# url = "http://127.0.0.1:5000/api/evaluation/"
# Use this api to require demo test service
url = "http://39.105.29.152:5000/api/evaluation/"
# This is the configureration yaml file
with open('./configs/fundus_evaluate_obs.yaml') as f:
    data = yaml.load(f)

script_path = data['script']
f = open(script_path,'r')
content = f.read()
# print(content)
encoded = base64.b64encode(content.encode('utf-8'))
encoded = str(encoded,'utf-8')
# print(encoded)
# 是否编码
data['script'] = encoded
decoded = str(base64.b64decode(encoded),'utf-8')
print(decoded)
headers = {'Content-Type': 'application/json;charset=UTF-8'}
# Post data and get response
response = requests.post(url,data=json.dumps(data),headers=headers)
# The response is expected to be a task id
print(response.text)
