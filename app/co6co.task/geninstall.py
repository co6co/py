# -*- coding:utf-8 -*-
from co6co import setupUtils

long_description = setupUtils.readme_content(__file__)
version = setupUtils.get_version(__file__)
print(f"生成{version}脚本...")
with open('install.bat', 'w+', encoding='utf-8') as f:
    f.write("@echo off\n")
    f.write("title 安装脚本\n")
    f.write("::设置编码为utf-8\n")
    f.write("chcp 65001\n")  # 设置utf-8 编码
    f.write("::echo 执行测试...\n")
    f.write("::python -m pytest tests/ -rA\n")
    f.write("::echo 测试完成.\n")
    f.write(f"echo resetup and reinstall 重新新版本:{version},并准备安装...\n")
    f.write("python -m compileall -q\n")
    f.write("echo 编译测试完成.\n")
    f.write("python setup.py sdist\n")
    f.write("::echo 任意键安装.\n")
    f.write("::pause\n")

    s = f"::pip uninstall co6co_task & pip install dist\\co6co_task-{version}.tar.gz"
    f.write(s)
print(f"生成{version}脚本完成.")
