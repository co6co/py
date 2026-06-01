@echo off
title 安装脚本
::设置编码为utf-8
chcp 65001
echo 执行测试...
python -m pytest tests/ -rA
echo 测试完成.
echo resetup and reinstall 重新新版本:0.1.3,并准备安装...
pause
python setup.py sdist & pip uninstall package_name & pip install dist\co6co-0.1.3.tar.gz