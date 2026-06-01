# -*- coding:utf-8 -*-
from setuptools import setup
from co6co import setupUtils

long_description = setupUtils.readme_content(__file__)
version = setupUtils.get_version(__file__)
packagesName, packages = setupUtils.package_name(__file__)
classifiers = setupUtils.get_classifiers()
setup(
    name=packagesName,
    version=version,
    description="wechat_ext",
    packages=packages,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    include_package_data=False, zip_safe=True,
    setup_requires=[ ], ## 构建时依赖 过时 且行为不稳定。
    # 依赖哪些模块 
    install_requires=["co6co>=0.1.0","wechatpy>=1.8.18"], # 运行时依赖
    # package_dir= {'utils':'src/log','main_package':'main'},#告诉Distutils哪些目录下的文件被映射到哪个源码
    author='co6co',
    author_email='co6co@qq.com',
    url="http://git.hub.com/co6co",
    data_file={
        ('', "*.txt"),
        ('', "*.md"),
    },
    package_data={
        '': ['*.txt', '*.md'],
        'bandwidth_reporter': ['*.txt']
    }
)