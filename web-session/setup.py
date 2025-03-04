from os import path
from setuptools import setup, find_packages


from co6co_web_session import versions
VERSION = versions.__version__

# read readmeFile contents
currentDir = path.abspath(path.dirname(__file__))
with open(path.join(currentDir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages=find_packages()
setup(
    name=packages[0],
    version=VERSION,
    description="web session 扩展",
    packages=packages,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=["Programming Language :: Python :: 3", "Programming Language :: Python :: 3.6"],
    include_package_data=True, zip_safe=True,
    # 依赖哪些模块
    install_requires=["co6co"],
    # package_dir= {'utils':'src/log','main_package':'main'},#告诉Distutils哪些目录下的文件被映射到哪个源码
    author='co6co',
    author_email='co6co@qq.com',
    url="http://github.com/co6co",
    data_file={
        ('', "*.txt"),
        ('', "*.md"),
    },
    package_data={
        '': ['*.txt', '*.md'],
        'bandwidth_reporter': ['*.txt']
    }
)
