@echo off
title 安装脚本
::设置编码为utf-8
chcp 65001
echo 执行测试...
python -m pytest tests/ -rA
echo 测试完成.
echo resetup and reinstall 重新新版本:0.1.1,并准备安装...
pause
python setup.py sdist & pip uninstall co6co.db_ext & pip install dist\co6co_db_ext-0.1.1.tar.gz