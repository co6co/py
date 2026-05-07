# -*- coding:utf-8 -*- 
from co6co import setupUtils

long_description = setupUtils.readme_content(__file__)
version = setupUtils.get_version(__file__) 
packagesName, packages = setupUtils.package_name(__file__)
print(f"生成{version}脚本...")
with open('install.bat', 'w+', encoding='utf-8') as f:
    f.write("@echo off\n")
    f.write("title 安装脚本\n")
    f.write("::设置编码为utf-8\n")
    f.write("chcp 65001\n") # 设置utf-8 编码
    f.write("echo 执行测试...\n")
    f.write("python -m pytest tests/ -rA\n")
    f.write("echo 测试完成.\n")
    f.write(f"echo 继续编译新版本{version}\n")
    f.write("pause\n") 
    
    s=f"python setup.py sdist\n"
    f.write(s)
    f.write("echo 是否安装?\npause\n") 
    f.write(f"echo resetup and reinstall 重新新版本:{version},并准备安装...\n")
    s=f"pip uninstall {packagesName} & pip install dist\\{packages[0]}-{version}.tar.gz"
print(f"生成{version}脚本完成.")
