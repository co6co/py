1. 重要
出现下列错误：
 ```
 Task <Task pending name='Task-11' 
 ```
 问题描述：1. 更新代码，保存后重启服务，经过1-2次出现该错误，查询未找到原因，后更新SQLALCHEMY 服务，待验证问题是否解决
 升级程序 SQLAlchemy：2.0.23 --> 2.0.25  2024-01-08


 pip install --upgrade SQLAlchemy 
  pip install --upgrade sanic
 pip install --upgrade sanic-ext