# -*- coding:utf-8 -*- 
from co6co import setupUtils

long_description = setupUtils.readme_content(__file__)
version = setupUtils.get_version(__file__) 
package_name,s= setupUtils.package_name(__file__)
name=s[0]

print(f"生成{package_name}_{version}脚本...,{s}")
with open('install.bat', 'w+', encoding='utf-8') as f:
    f.write("@echo off\n")
    f.write("title 安装脚本\n")
    f.write("::设置编码为utf-8\n")
    f.write("chcp 65001\n") # 设置utf-8 编码
    f.write("echo 执行测试...\n")
    f.write("python -m pytest tests/ -rA\n")
    f.write("echo 测试完成.\n")
    f.write(f"echo resetup and reinstall 重新新版本:{version},并准备安装...\n")
    f.write("pause\n")  
    s=f"python setup.py sdist & pip uninstall {package_name} & pip install dist\\{name}-{version}.tar.gz"
    f.write(s)

with open('upload.bat', 'w+', encoding='utf-8') as f:
    f.write("@echo off\n")
    f.write("title 上传安装包\n")
    f.write("::设置编码为utf-8\n")
    f.write("chcp 65001\n") # 设置utf-8 编码 
    f.write("pause\n")  
    s=f"twine upload dist\\{name}-{version}.tar.gz"
    f.write(s)
print(f"生成{package_name}_{version}脚本完成.")
