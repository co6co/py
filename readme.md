# 安装虚环境
// 创建的虚拟环境的存放的路径 C:\Users\Administrator\Envs
pip install virtualenv
pip install virtualenvwrapper-win # 扩展包（指令便捷）
#


virtualenv [虚拟环境名称] 
virtualenv venv

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