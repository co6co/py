@echo off
cd .
goto end

:end 
set /p file=pwdFile:
if /i "%file%" == "q" exit
 
python baseHttp.py -u http://192.168.1.222 -n admin -f "%file%"
goto :end