import requests
import json
import yaml
# Use this api to require local test service
# url = "http://127.0.0.1:5000/api/task/"
# Use this api to require demo test service
url = "http://123.60.209.79:5000/api/task/"
url = "http://localhost:5000/api/task/"
# This is the configureration yaml file
with open('../configs/local.yaml') as f:
    data = yaml.load(f)
headers = {'Content-Type': 'application/json;charset=UTF-8'}
# Post data and get response
response = requests.post(url,data=json.dumps(data),headers=headers)
# The response is expected to be a task id
print(response.text)
