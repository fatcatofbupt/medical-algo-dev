import requests
import json
import yaml
import base64
# Use this api to require local test service
# url = "http://127.0.0.1:5000/api/task/"
# Use this api to require demo test service
url = "http://39.105.29.152:5000/api/task/"
# This is the configureration yaml file
with open('./configs/fundus_create_obs.yaml') as f:
    data = yaml.load(f)
# print(data)
headers = {'Content-Type': 'application/json;charset=UTF-8'}
# Post data and get response
response = requests.post(url,data=json.dumps(data),headers=headers)
# The response is expected to be a task id
print(response.text)
