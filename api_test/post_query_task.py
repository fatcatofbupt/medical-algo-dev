import requests
'''
    Use this api to require local query service
    Change the tokens after test/ to a task id
'''
url = "http://127.0.0.1:5000/api/test/9054d1ea-1993-480c-b6a1-699e21c6663c"
''' 
    Use this api to require demo query service
    Change the tokens after test/ to a task id
'''
# url = "https://123.60.209.79:5000/api/test/cd0d8084-066d-419e-aa9c-873404783bef"
headers = {'Content-Type': 'application/json;charset=UTF-8'}
response = requests.post(url,headers=headers)
''' 
The response is expected to be a json-like file
For example：
    {
        "state": "SUCCESS",
        "current": 0,
        "success": 1,
        "total": 2,
        "fail_to_read": [],
        "fail_to_attack": [],
        "status": "Task Complete!"
    }
'''
print(response.text)