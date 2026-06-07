cd db-ext
python geninstall.py
call install.bat
echo db_ext 执行完毕.

cd ..\web
python geninstall.py
call install.bat

cd ..\web-db
python geninstall.py
call install.bat

cd ..\web-session
python geninstall.py
call install.bat

cd ..\app\co6co.task
python geninstall.py
call install.bat

cd ..\permissions
python geninstall.py
call install.bat
