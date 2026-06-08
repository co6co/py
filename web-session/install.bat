@echo off
title 安装脚本
::设置编码为utf-8
chcp 65001
echo 执行测试...
python -m pytest tests/ -rA
echo 测试完成.
echo resetup and reinstall 重新新版本:0.1.0,并准备安装...
pause
python setup.py sdist & pip uninstall co6co.web_session & pip install dist\co6co_web_session-0.1.0.tar.gz