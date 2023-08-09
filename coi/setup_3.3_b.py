from distutils.core import setup
from setuptools import find_packages 

setup(
    name="co6co",
    version='0.0.1',
    description="基础模块" ,
    #py_modules=['log'],
    author='co6co',
    author_email ='co6co@qq.com',
    url="http://git.hub.com/co6co",
    install_requires=['loguru'], #依赖哪些模块
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Utilities'
    ],
    packages =find_packages('src'), #告诉Distutils需要处理那些包（包含__init__.py的文件夹）
    package_dir= {'':'src'},#告诉Distutils哪些目录下的文件被映射到哪个源码
    include_package_data = True #package_data =  {'src':['macpy/oui_3.dict', 'macpy/oui_3.dict']} #通常包含与包实现相关的一些数据文件或类似于readme的文件。如果没有提供模板，会被添加到MANIFEST文件中。
)