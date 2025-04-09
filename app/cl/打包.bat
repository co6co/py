@echo off
cls
::pyinstaller --onefile .\wordAndExcell.py
set /p icon=ICON File Path:
set /p pyFile=Py File Path:

::--windowed 参数的作用是隐藏终端窗口（控制台）
set /p other =other(--windowed):
echo  确保主脚本中有正确的入口点函数（通常是 if __name__ == "__main__": 块）
::pyinstaller --onefile --windowed --icon=%icon% %pyFile%
:: 增加资源文件 --add-data "tracerite/style.css;tracerite" 
:: 隐式依赖 --hidden-import requests

:: 生产.spec 文件， pyinstaller --onefile your_script.py --specpath .
:: pyinstaller your_script.spec
pyinstaller %other%--onefile --icon=%icon% %pyFile%
pause
