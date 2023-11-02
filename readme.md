# 安装虚环境
```
// 创建的虚拟环境的存放的路径 C:\Users\Administrator\Envs
pip install virtualenv
pip install virtualenvwrapper-win # 扩展包（指令便捷）
#
 
virtualenv [虚拟环境名称] 
virtualenv venv

virtualenv venv --system-site-packages # 包含主环境包

#如果不想使用系统的包,加上–no-site-packeages参数
virtualenv  --no-site-packages 创建路径名
> cd venv
> .\Scripts\activate.bat

workon #列出虚拟环境列表
workon [venv] #切换环境
deactivate
rmvirtualenv venv
 
python<version> -m venv <virtual-environment-name>
pip freeze > requirements.txt  # 列出项目依赖 
pip install -r requirements.txt
```
# 安装包和发布包
```
# 本地安装
python setup.py sdist & pip uninstall co6co & pip install dist\co6co-0.0.1.tar.gz
# 检查一下包
twine check dist/*
#发布项目 
twine upload dist/*
```

```
// C:\Users\Administrator\.pypirc
[distutils]
index-servers=pypi
repository= https://pypi.org/legacy/
[pypi]
username= __token__
password= pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXx
```
```
-----BEGIN PGP MESSAGE-----
Version: GnuPG v1

hQEMA4fLSGZcsDhlAQf+J9VpomM69yZo3yA0gLJzhK1GP2nXOelZpb7g5kCCREC2
vvC1vSKJf3aZcbsmOLUWoG0F3iIC6agRhVWPs31Klz06TTvFfjqZxiBxhMQ2LXli
wJquFqGCp9A7oS9FJEOAONhWiPznAezGKm/VJqp7XxFy/AwS27u2fb07alWa7uor
DbPL8PlfARcypusFW2k31GBJz/4z1JTmIOVSyMQnSti6yN1EEOIYkGZwj/sxAmiu
v3/Qkuvp4ua+hDoppNp5lDT8JEIpFb2Xby0dwpbkq9gBvcACzZCH7QGORpkzQ80Y
8GRUxkNss9PxRkuRlB846JW98llNclPUm2pjO2wsNdLAIAHoRnJqpXPClph61W4Y
lf3/ZxSd6myeQJAVBM6DPJ3/5V90TN5xD4jnnPlupJ1t+M+Dkynxj20v0a18tZBZ
fXQEqyCnRWAwFLgYlZ8kfDu5WAdkPnm9fB0gD1HLZAx2CG78pHvLl+mrjxlPEdU/
wIzTY2IOA07P5fidNXDnl5ZcNeVXMrV2O1qQHg5GZMN4tiwl2GQ2FR9eyMwAPA92
DpMXbv2Hf5RmM6NtS0BkDfcn1HQC8tdZ9nWhowBMSxr2oDYXkceXzWMy1/+os7ah
I1gKs2NXGU0efjbxLFCbjiLB
=ZtVM
-----END PGP MESSAGE-----

```

# Pypi 镜像
## 1. 临时使用
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```
## 2. 设为默认
pip >=10.0.0
```
python -m pip install --upgrade pip
## 使用镜像升级
## python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
## 3. (多镜像)[https://mirrors.cernet.edu.cn/list/pypi]
```
pip config set global.extra-index-url "<url1> <url2>..."
```
## 4. 其他设计镜像方式
```
#PDM 通过如下命令设置默认镜像： 
pdm config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple

#Poetry 通过以下命令设置默认镜像：
poetry source add --priority=default mirrors https://pypi.tuna.tsinghua.edu.cn/simple/
#通过以下命令设置次级镜像：
poetry source add --priority=secondary mirrors https://pypi.tuna.tsinghua.edu.cn/simple/
```