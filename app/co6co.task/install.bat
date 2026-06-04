@echo off
title 安装脚本
::设置编码为utf-8
chcp 65001
echo 执行测试...
python -m pytest tests/ -rA
echo 测试完成.
echo resetup and reinstall 重新新版本:0.1.4,并准备安装...
pip index versions co6co.task
pause
python setup.py sdist & pip uninstall co6co.task & pip install dist\co6co_task-0.1.4.tar.gz