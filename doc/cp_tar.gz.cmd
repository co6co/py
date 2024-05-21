@echo off
set src=H:\Work\Projects\html\py
set co6co=\coi\dist\co6co-0.0.5.tar.gz
set db_ext=\db-ext\dist\co6co.db_ext-0.0.5.tar.gz
set sanic_ext=\web\dist\co6co.sanic_ext-0.0.2.tar.gz
set web_db=\web-db\dist\co6co.web_db-0.0.5.tar.gz
set permissions=\app\permissions\dist\co6co.permissions-0.0.3.tar.gz
copy "%src%%co6co%" .\
copy "%src%%db_ext%" .\
copy "%src%%sanic_ext%" .\
copy "%src%%web_db%" .\
copy "%src%%permissions%" .\
pause