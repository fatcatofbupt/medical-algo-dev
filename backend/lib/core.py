import base64
import sys
sys.path.append('./')
import runpy
import copy
from backend.lib.datasets import DatasetFactory
from backend.lib.algorithms import TestFactory
from backend.lib.models import ModelFactory
from backend.lib.newdata_eval import NewdataEvalFactory

import yaml
import requests
import json
import zipfile
import torch
import time
import os
import argparse
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

class StepCallback(object): # 步骤回调函数。可以提供进度
    def __init__(self):
        super(StepCallback, self).__init__()
        # self.count = 0


    def __call__(self, step, total):
        print('step callback', f'{step+1}/{total}')


class StorageCallback(object): # 存储回调函数。用于导出新生成的数据

    def __init__(self):
        super(StorageCallback, self).__init__()


    def __call__(self, input_raw, label, meta):
        print('storage callback', meta)

def get_method_info():
    return TestFactory.show_methods()

def get_dataset_info():
    return DatasetFactory.show_datasets()

def get_model_info():
    return ModelFactory.show_models()

# 扰动测试评价
class BenchmarkerCore(object):
    def __init__(self):
        super(BenchmarkerCore, self).__init__()

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
        self.bucket = 'adversial-attack-1307678320'
        token = response['Credentials']['Token']
        scheme = 'https'
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
        
        self.client = CosS3Client(config)

    def download_model(self,key,download_dir,model_name):
        resp = self.client.get_object(Bucket=self.bucket,Key=key)
        print("RESP:",resp)
        fp = resp['Body'].get_raw_stream()
        file_dir = os.path.join(download_dir,model_name + '.pth')
        with open(file_dir,'wb') as f:
            f.write(fp.read())
        return file_dir

    def zip_and_upload(self,data_dir,zip_dir,upload_name):
        zip = zipfile.ZipFile(zip_dir,'w')
        print("Start Zipping...")
        shortest = len(data_dir.split('/')) + 1
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root,file)
                if 'model' in file_path:
                    continue
                if len(file_path.split('/'))> shortest:
                    zip.write(file_path,os.path.join(file_path.split('/')[-2],file))
                else:
                    zip.write(file_path,file)
        zip.close()
        print("Zip Finished!")
        print("Start Uploading...")
        with open(zip_dir,'rb') as fb:
            response = self.client.put_object(
                Bucket=self.bucket,
                Body=fb,
                Key='generated_dataset/' + upload_name, # 上传后的文件名称，可以包含文件路径
                StorageClass='Standard',
            )
        print(response['ETag'])
        print("Upload Finished!")
        
    def download_data_unzip(self,zip_dir,download_dir):
        resp = self.client.list_objects(Bucket=self.bucket,Prefix=zip_dir,MaxKeys=1000)
        print(zip_dir)
        print(resp)
        file_dir = ''
        for content in resp['Contents']:
            key = content['Key']
            if key == zip_dir:
                resp = self.client.get_object(Bucket=self.bucket,Key=key)
                fp = resp['Body'].get_raw_stream()
                print(key)
                file_dir = os.path.join(download_dir,key.split('/')[-1])
                with open(file_dir,'wb') as f:
                    f.write(fp.read())
        if zipfile.is_zipfile(file_dir) and file_dir!='':
            with zipfile.ZipFile(file_dir, 'r') as zipf:
                dataset_dir = os.path.join(download_dir,key.split('/')[-1].strip('.zip'))
                os.makedirs(dataset_dir,exist_ok=True)
                zipf.extractall(dataset_dir)
            print("Successfully Downloaded and Unzipped!")
        else:
            print("Failed to Download!")
        print("Download Path:",os.path.join(download_dir,key.split('/')[-1].strip('.zip')))
        return os.path.join(download_dir,key.split('/')[-1].strip('.zip'))

    def download_data(self,root_dir,download_dir,data_len):
        resp = self.client.list_objects(Bucket=self.bucket,Prefix=root_dir,MaxKeys=1000)
        # img_dir = os.path.join(root_dir,'images')
        # os.makedirs(os.path.join(download_dir,'images'),exist_ok=True)
        label_dir = root_dir
        names = []
        json_downloaded = False
        for content in resp['Contents']:
            key = content['Key']
            # print("Key",key)
            if img_dir in key:
                if '.' not in key:
                    continue
                names.append(key)
                resp = self.client.get_object(Bucket=self.bucket,Key=key)
                fp = resp['Body'].get_raw_stream()
                file_dir = os.path.join(download_dir,'images',key.rstrip('/').split('/')[-1])
                with open(file_dir,'wb') as f:
                    f.write(fp.read())
            elif label_dir in key:
                if '.json' not in key:
                    continue
                resp = self.client.get_object(Bucket=self.bucket,Key=key)
                fp = resp['Body'].get_raw_stream()
                file_dir = os.path.join(download_dir,key.rstrip('/').split('/')[-1])
                with open(file_dir,'wb') as f:
                    f.write(fp.read())
                json_downloaded = True
            if len(names) == data_len and json_downloaded == True:
                break

    def _run(self, dataset, model, test_algorithm, evaluation):
        '''
        dataset
        model
        test_algorithm
        evaluation
        '''
        records = []
        total = len(dataset)
        for step, (input_raw, label, meta) in enumerate(dataset):
            if not (test_algorithm is None):
                input_raw, label, meta = test_algorithm.run(input_raw, label, meta, model, self._device)
                if not (self._storage_hook is None):
                    self._storage_hook(input_raw, label, meta)
            record = evaluation.predict(input_raw, label, meta, model)
            records += [record]
            self._step_hook(step, total)
        predict_list = records
        result = evaluation.criteria(predict_list)
        return result


    def create_dataset(self, params, use_obs=False):
        # print("Hi There")
        step_hook = StepCallback()
        self._step_hook = step_hook

        model_name = params.pop('model_name')
        model_params = params.pop('model_params')
        device = model_params.get('device','cpu')
        self._device = device

        newdata_evals = params.get('newdata_evals',[])
        dataset_name = params.pop('dataset_name')
        dataset_params = params.pop('dataset_params')

        test_name = params.pop('test_name')
        test_params = params.pop('test_params')
        out_dir = params.pop('out_dir')
        #OBS Settings
        if use_obs:
            tmp_dir = params.pop('tmp_dir')
            model_tmp_dir = params.pop('model_tmp_dir')
            zip_file = params.pop('zip_file')
            upload_name = params.pop('upload_name')
            obs_dataset_dir = dataset_params['root_dir']
            obs_model_dir = model_params['path']
            dataset_params['root_dir'] = self.download_data_unzip(obs_dataset_dir,tmp_dir)
            model_params['path'] = self.download_model(obs_model_dir,model_tmp_dir,'model')
            if test_name == 'algorithm_fundus_cyclegan':
                print('Using Cyclegan...')
                test_params['a_path'] = self.download_model(test_params['a_path'],model_tmp_dir,'G_A')
                test_params['b_path'] = self.download_model(test_params['b_path'],model_tmp_dir,'G_B')
            # model_params['path'] = './models/lung_ssd.pth'
            dataset_out_params = copy.deepcopy(test_params)
            dataset_out_params['root_dir'] = out_dir
            # dataset_params['download_dir'] = tmp_dir
            # model_params['download_dir'] = model_tmp_dir
            # dataset_params['bucket'] = params.pop('bucket')
            # dataset_params['client'] = params.pop('client')
        else:            
            dataset_out_params = copy.deepcopy(test_params)
            dataset_out_params['root_dir'] = out_dir

        dataset = DatasetFactory.create(dataset_name,**dataset_params)
        model = ModelFactory.create(model_name,**model_params)

        test_algorithm = TestFactory.create(test_name,**test_params)
        
        newdata_eval_callbacks = []
        for newdata_eval in newdata_evals:
            newdata_eval_name = newdata_eval['newdata_eval_name']
            newdata_eval_params = newdata_eval.get('newdata_eval_params',{})
            newdata_eval = NewdataEvalFactory.create(newdata_eval_name,**newdata_eval_params)
            if not (newdata_eval is None):
                newdata_eval_callbacks += [newdata_eval]
        
        total = len(dataset)
        dataset_info = dataset.init(dataset_out_params)
        scores_individual = []
        total = len(dataset)
        print("Length of Dataset:",total)
        for step, (input_ori, label_ori, meta_ori) in enumerate(dataset):
            if not (test_algorithm is None):
                input_new, label_new, meta_new = test_algorithm.run(input_ori, label_ori, meta_ori, model, self._device)
                dataset_info = dataset.add(input_new, label_new, meta_new, dataset_info)
                score_individual = meta_new.copy()
                for newdata_eval in newdata_eval_callbacks:
                    score = newdata_eval.single(input_ori, label_ori, meta_ori, input_new, label_new, meta_new)
                    score_individual.update(score)
                scores_individual += [score_individual]
            self._step_hook(step, total)
        score_summary = {}
        for newdata_eval in newdata_eval_callbacks:
            score = newdata_eval.summary(scores_individual)
            score_summary.update(score)
        if use_obs:
            self.zip_and_upload(tmp_dir,zip_file,upload_name)
            dataset_info['zipfile_name'] = upload_name
        dataset_output_params = {
            'dataset_name': dataset_name,
            'dataset_params': dataset_info,
            'dataset_evaluation': {
                'individual': scores_individual,
                'summary': score_summary,
            },
        }
        result = dataset_output_params
        print("Result:",result)
        return str(result)


    def evaluate_only(self, params,use_obs=False):
        step_hook = StepCallback()
        storage_hook = StorageCallback()

        model_name = params.pop('model_name')
        model_params = params.pop('model_params')
        device = model_params.get('device','cpu')

        dataset_name = params.pop('dataset_name')
        dataset_params = params.pop('dataset_params')

        self._step_hook = step_hook
        self._storage_hook = storage_hook
        self._device = device

        if use_obs:
            tmp_dir = params.pop('tmp_dir')
            model_tmp_dir = params.pop('model_tmp_dir')
            obs_dataset_dir = dataset_params['root_dir']
            obs_model_dir = model_params['path']
            dataset_params['root_dir'] = self.download_data_unzip(obs_dataset_dir,tmp_dir)
            model_params['path'] = self.download_model(obs_model_dir,model_tmp_dir,'model')

            evaluation_script = params.pop('script')
            script_dir = params.pop('script_tmp_dir')
            evaluation_script_decoded = str(base64.b64decode(evaluation_script),'utf-8')
            print(evaluation_script_decoded)
            with open(script_dir,'w') as f:
                f.write(evaluation_script_decoded)
        else:
            script_dir = params.pop('script')
        file_globals = runpy.run_path(script_dir)

        # print(len(file_globals.keys()))
        for key in file_globals:
            # print(key)
            if not ('__' in key):
                evaluation_entry = key
        print(evaluation_entry)
        # assert(len(file_globals.keys()) == 1)
        # evaluation_entry = file_globals.keys()[0]
        # for key in file_globals:
        #     print(key)
        EvaluationCls = file_globals[evaluation_entry]

        # #benchmarker = BenchmarkerCore(step_hook, storage_hook, device)
        model = ModelFactory.create(model_name,**model_params)
        dataset = DatasetFactory.create(dataset_name,**dataset_params)

        evaluation = EvaluationCls()

        result = self._run(dataset, model, None, evaluation)
        # print("Result:",result)
        return result




def make_new_params(params,evaluate_only = False):
    params_new = copy.deepcopy(params)
    now_time = str(time.time()).replace('.','')
    dataset_info = params['dataset']
    dataset_name = dataset_info['name']
    dataset_params = dataset_info['params']
    model_info = params['model']
    model_name = model_info['name']
    model_params = model_info['params']
    
    params_new['dataset_name'] = dataset_name
    params_new['dataset_params'] = dataset_params
    params_new['model_name'] = model_name
    params_new['model_params'] = model_params

    if not evaluate_only:
        out_dir = 'test/tmp_out/' + now_time
        test_info = params['algorithm']
        test_name = test_info['test']['name']
        test_params = test_info['test']['params']
        params_new['test_name'] = test_name
        params_new['test_params'] = test_params
        params_new['out_dir'] = out_dir
        os.makedirs(out_dir,exist_ok=True)

    return params_new


def test_create_dataset(create_path):
    with open(create_path) as f:
        params = yaml.load(f)
    print(params)
    algorithm = params.pop('algorithm')
    tests = algorithm['tests']
    for test in tests:
        params['algorithm'] = {}
        params['algorithm']['test'] = test
        params_new = make_new_params(params,evaluate_only=False)
        benchmarker = BenchmarkerCore()
        print(params)
        result = benchmarker.create_dataset(params_new)
        print(result)

def test_evaluate_only(evaluate_path):
    with open(evaluate_path) as f:
        params = yaml.load(f)
    params_new = make_new_params(params,evaluate_only=True)
    benchmarker = BenchmarkerCore()
    print(params)
    result = benchmarker.evaluate_only(params_new)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_path',default='',help='The input parameters of image creation')
    parser.add_argument('--evaluate_path',default='',help='The input parameters of image evaluation')
    args = parser.parse_args()

    # args.create_path = 'configs/clinical_generalize.yaml'
    # args.create_path = 'configs/clinical_create.yaml'
    # args.evaluate_path = 'configs/clinical_evaluate.yaml'
    
    if args.create_path != '':
        test_create_dataset(args.create_path)
    else:
        print()
        print("Empty create path, skip creating dataset")
        print()
    if args.evaluate_path != '':
        test_evaluate_only(args.evaluate_path)
    else:
        print()
        print("Empty evaluate path, skip evaluating model")
        print()

