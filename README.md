## Docker封装
### 1. 依赖
* docker
	MAC:
		安装: 暂时忘了，
		版本:`Docker version 20.10.5, build 55c4c88`
* docker-compose
	MAC:
		安装: `brew install docker-compose`
		版本: `docker-compose version 1.29.0, build 07737305 `

 
### 2. 文件结构
```

.
├── Dockerfile.celery.yml					-----celery 任务队列 dockerfile
├── Dockerfile.web.yml						-----Flask  dockerfile
├── README.md
├── __pycache__
│   └── attack_tools.cpython-38.pyc
├── api
│   ├── app
│   └── standalone.py
├── api_test
│   ├── post_test_adv.py
│   ├── post_test_query.py
│   └── test_local_docker.py
├── backend
│   └── lib
├── configs
│   ├── local.yaml
│   ├── server.yaml
│   └── simple_test.yaml
├── docker-compose.yml						----- docker compose 入口文件
├── models							----- 模型文件所在目录
│   └── jit_module_448_cpu.pth
├── requirements.celery.txt					-----celery任务队列的依赖，目前是跟Flask混在一起，建议分开
├── requirements.txt						-----原来的python依赖文件
├── requirements.web.txt					-----Web 服务的依赖，目前是跟celery混在一起，建议分开
├── test
│   ├── binary_criteria.py
│   └── fundus
├── try_upload.py
├── 测试算法接口文档.md
└── 测试算法部署文档.md

10 directories, 20 files
```

### 3. USAGE
1. 进入项目目录，构建镜像并启动容器: `docker-compose up -d --build`, docker相关信息查看方式：
	1. 查看容器：`docker ps -a `
	```
	$ docker  ps -a
	CONTAINER ID   IMAGE                  COMMAND                  CREATED        STATUS                    PORTS                    NAMES
	48ee54559801   retina_attack_web      "python api/standalo…"   24 hours ago   Up 22 minutes             0.0.0.0:5000->5000/tcp   retina_attack_web_1
	d20b0b2cfa1c   retina_attack_worker   "celery -A api.stand…"   24 hours ago   Up 19 minutes                                      retina_attack_worker_1
	039514c5ef51   redis:alpine           "docker-entrypoint.s…"   24 hours ago   Up 20 minutes             0.0.0.0:6379->6379/tcp   retina_attack_redis_1
	```
	2. 查看日志：`docker logs -f retina_attack_web_1`
		```
		$ docker  logs -f retina_attack_web_1
		 * Serving Flask app "api.app" (lazy loading)
		 * Environment: production
		   WARNING: This is a development server. Do not use it in a production deployment.
		   Use a production WSGI server instead.
		 * Debug mode: on
		 * Running on all addresses.
		   WARNING: This is a development server. Do not use it in a production deployment.
		 * Running on http://172.18.0.4:5000/ (Press CTRL+C to quit)
		 * Restarting with stat
		 * Debugger is active!
		 * Debugger PIN: 102-771-169
		 * Serving Flask app "api.app" (lazy loading)
		 * Environment: production
		   WARNING: This is a development server. Do not use it in a production deployment.
		   Use a production WSGI server instead.
		 * Debug mode: on
		 * Running on all addresses.
		   WARNING: This is a development server. Do not use it in a production deployment.
		 * Running on http://172.18.0.2:5000/ (Press CTRL+C to quit)
		 * Restarting with stat
		 * Debugger is active!
		 * Debugger PIN: 292-171-720
		 * Detected change in '/web/api_test/post_test_adv.py', reloading
		 * Restarting with stat
		 * Debugger is active!
		 * Debugger PIN: 292-171-720
		 * Detected change in '/web/api_test/post_test_adv.py', reloading
		 * Restarting with stat
		 * Debugger is active!
		 * Debugger PIN: 292-171-720
		172.18.0.1 - - [17/Oct/2021 08:39:52] "POST /api/task/ HTTP/1.1" 200 -

		```
	3. 重启某个容器: `docker restart retina_attack_web_1`
	4. 登录容器: `docker exec -ti retina_attack_web_1 /bin/bash` 
2. 确定各个docker都已启动，测试服务可用性: `cd api_test; python test_local_docker.py`
### 4. ToDo
1. 代码文件结构可以优化下，web(Flask)和celery可以分开，基于新的代码结构，单独构建`web/requirements.yml`和`celery/requirements.yml`,这样可以降低docker镜像体积。新的结构比如：
	```
	├── web
	│   ├── Dockerfile
	│   ├── Dockerfile.dev
	│   ├── app.py
	│   ├── requirements.txt
	│   └── worker.py
	├── celery
	│   ├── Dockerfile
	│   ├── Dockerfile.dev
	│   ├── requirements.txt
	│   └── tasks.py
	├── docker-compose.development.yml
	└── docker-compose.yml
	```
2. 抽取代码中的配置信息到配置文件，包括各种文件路径等

