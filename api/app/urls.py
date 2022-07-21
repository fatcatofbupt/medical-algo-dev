from flask_restful import Api
from .server import DeleteEvaluation, Evaluation, Hello, QueryEvaluation,Task,QueryTest,QueryTask,DeleteTask,GetInfo, TaskReport #导入路由的具体实现方法
from flask import Blueprint

#创建一个蓝图对象
assets_page = Blueprint('assets_page', __name__)
#在这个蓝图对象上进行操作,注册路由
api = Api(assets_page)

#注册路由
api.add_resource(Hello,'/')
api.add_resource(Task,'/api/task/')
api.add_resource(Evaluation,'/api/evaluation/')
api.add_resource(QueryTest,'/api/test/<_id>')
api.add_resource(TaskReport,'/api/task/<_id>/report/')
api.add_resource(QueryTask,'/api/task/<_id>')
api.add_resource(QueryEvaluation,'/api/evaluation/<_id>')
api.add_resource(GetInfo,'/api/get_info/')
api.add_resource(DeleteTask,'/api/del_task/<_id>')
api.add_resource(DeleteEvaluation,'/api/del_evaluation/<_id>')