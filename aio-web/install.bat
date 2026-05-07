@echo off
title 安装脚本
::设置编码为utf-8
chcp 65001
echo 执行测试...
python -m pytest tests/ -rA
echo 测试完成.
echo 继续编译新版本0.0.1
pause
python setup.py sdist
echo 是否安装?
pause
echo resetup and reinstall 重新新版本:0.0.1,并准备安装...
