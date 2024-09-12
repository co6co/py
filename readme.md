# 安装虚环境
## 1. virtualenv
```
// 创建的虚拟环境的存放的路径 C:\Users\Administrator\Envs

pip install virtualenv
pip install virtualenvwrapper-win # 扩展包（指令便捷）
#
 
virtualenv [虚拟环境名称] 
virtualenv venv


virtualenv venv --system-site-packages # 包含主环境包
virtualenv /envs/test --python=python3.9

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
<<<<<<< HEAD


```
## 2. conda
conda create --name env_name python=3.11
conda env list
conda activate env_name
conda deactivate

conda list # 包管理
### 2.1 查看Python 所在路径
where python #windows
which python # linux or macOS



# 包管理
=======
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
pypi Data:
```
-----BEGIN PGP MESSAGE-----
Version: GnuPG v1

hQEMA4fLSGZcsDhlAQgApJWIP/xpbJT2v/71XvYgCx3XmfKTbIroElD0b3IbhgVr
GSA+Xr9m/pWAWBa3QJo7zEt6witHNEBEyD58M/vthh5IdZpe58q7yroioKJZuq9k
RfswgSL8fpTTGvBPK8uW0V18+NnnQQyKYF6r8bzw9u6WoF+IGupD3yfHA1EY8EcZ
zb0ZsZZyEQZbWzM5pYN8uHeP0Uqp/ghTSuqjPD8OWByiSnU/YXQ6YFC/vBo2pvd1
VKmLPVIqUuDCB47MksAl7+ABh44dA16wLeAnGMtrtLprQOzpvRYr322VHtKb+Jg8
rB+AcNNiRkcEZ6Trq6eMlADyvhiI+Wsgw8xiDlTSNNLAQgEK3rN1WK8r4QSQL99D
dlJZJKctGUpV4RlXiiQVg3x2hcLXMi5HGfmH7q9UGdyksirrfSWq8SivK/VHpfMg
g+Zt0ERQDpOH4LKUCo9XZio6Cma1qL2a1c6U9dNIkujpGRNbrnMQQX/ppZifAFfl
PGIf5I41hj1FklIpAp/7mavBjEPJO6vMEjbjYb76QEWgl7p+PHPvWrhhXXNTj0pj
UBLHhnxqN6ard0yhA6AzrdJU1OO0pz73kONdSntnYxx6smHP4vDGtgGu3YArd7vw
PsemSlrpirq7Ou2JxrRubpb1q6I7DuSfgpompA6gG+qhXGqkFNIi2aelM5//KSfr
Sazv2w==
=gpR9
-----END PGP MESSAGE-----


PyPI recovery codes:
-----BEGIN PGP MESSAGE-----
Version: GnuPG v1

hQEMA4fLSGZcsDhlAQgAsZBX+ofUxQC4KcAZS2VibS0U9QudDiN9yRyL1RYcIU74
o37Bsjvq6BTon1af42Hu+z1sC6qBEB+P7F1RyFTj7E+P+AMjoIRubex9jY0zNDPr
INr/Gg3TtAK1PlfI91ASP1h1Yo/jONHTi0gViMlkKrX0TPyZad9dMsQg09wYDYsV
9nq0mJ+N/o9PKpPYRUdfIgCZHVm3sIZ3wqamrabD2dlnHTVtlLwKZTt/bak+TcR+
dymxes0cjMEN41U5UrdN1i+wvrUDlawO0SHySoiLjs4mZJlqLv93KKYlC3yt8/nz
Y9lANdrnHr5LwY0FqfvYBESwMC44LMMdY/H+no6HGdK2ATkViFF/UtkKIrRBNm7f
shG5EvLeZUrA2jetFwLmR3VdbyCqC6JVfbpuQsJnnBLtoNBj4OoqVq0YsScX6q0x
rZ4lc2QSUPIWSOowVk2/E7zEReo5UiUBXqvp8qLDx3F5OLK/FDH0E4pmPxtsY8kb
hhhrsD0l10/UsfOn29f6ur4aYYruOCtX9w46g7OpYzWdGypAy8kzB+aRnt3790gS
kgdkx3O47yh+IvB7lNceUhAji/TzfoE=
=ReA2
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