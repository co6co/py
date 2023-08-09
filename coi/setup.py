from setuptools import setup, find_packages 
setup (
    name="co6co",
    version='0.0.1',
    description="基础模块" ,
    packages=find_packages(),
    #package_dir= {'':'src'},#告诉Distutils哪些目录下的文件被映射到哪个源码
    author='co6co',
    author_email ='co6co@qq.com',
    url="http://git.hub.com/co6co",
    install_requires=['loguru'], #依赖哪些模块
)