from flask import Flask
from .celery import celery_app
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    
    #CORS跨域：FLask下如何解决跨域问题
    CORS(app,supports_credentials=True) 
    
    celery_app.conf.update({"broker_url": 'redis://172.17.35.125:6379/0',
                            "result_backend": 'redis://172.17.35.125:6379/0', })
    # 导入创建的蓝图
    from .urls import assets_page
    # 注册这个蓝图对象
    app.register_blueprint(assets_page)
    return app
