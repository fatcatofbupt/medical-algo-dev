from logging import info
from flask import Flask,request,jsonify
import requests
from flask_restful import  Api, Resource
import json
from .celery import test_all,evaluate,test_and_create
from backend.lib.core import get_method_info,get_dataset_info,get_model_info
# app = Flask(__name__)
# CORS(app,supports_credentials=True)
# api = Api(app)

class Hello(Resource):
    def get(self):
        return 'Hello World!'

class Task(Resource):
    def get(self):
        return "Robustness Test Task"
    def post(self):
        params = json.loads(request.data)
        task = test_all.apply_async(args=[params],ignore_result=False)
        task_id = task.id
        test_ids = task.get()
        return {'task_id': task_id,'sub_ids': test_ids}

class Evaluation(Resource):
    def get(self):
        return "Evaluation of Models by the generated dataset"
    def post(self):
        params = json.loads(request.data)
        task = evaluate.apply_async(args=[params],ignore_result=False)
        task_id = task.id
        return {'task_id':task_id}

class GetInfo(Resource):
    def get(self):
        method_info = get_method_info()
        dataset_info = get_dataset_info()
        model_info = get_model_info()
        info_dict = {'model':model_info,'dataset':dataset_info,'algorithm': method_info}
        with open('./info.json','w') as f:
            json.dump(info_dict,f,ensure_ascii=False)
        return jsonify(info_dict)

class TaskReport(Resource):
     def get(self,_id):
        task = test_all.AsyncResult(_id)
        if task.state == 'SUCCESS':
            state = "PENDING"
            task_ids = task.result
            finished = 0
            total = len(task_ids)
            results = []
            for task_id in task_ids:
                test = test_and_create.AsyncResult(task_id)
                if test.state == "SUCCESS":
                    finished += 1
                    results.append(test.result)
                elif test.state != "PENDING":
                    state = "STARTED"
            progress = "{}/{} task(s) finished".format(finished,total)
            if finished == total:
                state = "SUCCESS"
            response = {
                'state':state,
                'results':results
            }
        else:
            response = {
                'state': task.state,
                'progress':"FAILURE"
            }
        return response

class QueryTask(Resource):
    def get(self,_id):
        task = test_all.AsyncResult(_id)
        if task.state == 'SUCCESS':
            state = "PENDING"
            task_ids = task.result
            finished = 0
            total = len(task_ids)
            results = []
            for task_id in task_ids:
                test = test_and_create.AsyncResult(task_id)
                if test.state == "SUCCESS":
                    finished += 1
                    results.append(test.result)
                elif test.state != "PENDING":
                    state = "STARTED"
            progress = "{}/{} task(s) finished".format(finished,total)
            if finished == total:
                state = "SUCCESS"
            response = {
                'state':state,
                'finished': finished,
                "pending": total - finished,
                'progress':progress,
            }
        else:
            response = {
                'state': task.state,
                'progress':"FAILURE"
            }
        return response

class QueryTest(Resource):
    def get(self,_id):
        task = test_and_create.AsyncResult(_id)
        if task.state == 'PENDING':
            #job did not start yet
            response = {
            'state': task.state,
            'result':{}
            }
        elif task.state != 'FAILURE':
            response = {
            'state': task.state,
            'result': task.result
            }
            # if 'result' in task.info:
            #     response['result'] = task.info['result']
        else:
            # something went wrong in the background job
            response = {
            'state': task.state,
            'status': str(task.info)  # this is the exception raised
            }
        return response

class QueryEvaluation(Resource):
    def get(self,_id):
        task = evaluate.AsyncResult(_id)
        if task.state == 'PENDING':
            #job did not start yet
            response = {
            'state': task.state,
            'result':{}
            }
        elif task.state != 'FAILURE':
            response = {
            'state': task.state,
            'result': task.result
            }
            # if 'result' in task.info:
            #     response['result'] = task.info['result']
        else:
            # something went wrong in the background job
            response = {
            'state': task.state,
            'status': str(task.info)  # this is the exception raised
            }
        return response


class DeleteTask(Resource):
    def post(self,_id):
        task = test_all.AsyncResult(_id)
        task_ids = task.result
        for task_id in task_ids:
                test = test_and_create.AsyncResult(task_id)
                test.forget()
        task.forget()
        return {'code':200,'status':'Task Deleted!'}

class DeleteEvaluation(Resource):
    def post(self,_id):
        task = evaluate.AsyncResult(_id)
        task.forget()
        return {'code':200,'status':'Task Deleted!'}


# api.add_resource(Hello, '/')
# api.add_resource(NormalAttack, '/api/norm')
# api.add_resource(AdversarialAttack, '/api/adv')

# if __name__ == '__main__':
#     server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
#     server.serve_forever()
#     app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000,debug=True)


