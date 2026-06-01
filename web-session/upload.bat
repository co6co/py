@echo off
title 上传安装包
::设置编码为utf-8
chcp 65001
pause
twine upload dist\co6co_web_session-0.0.4.tar.gz