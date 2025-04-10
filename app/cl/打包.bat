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


::打包2  C
::::::  pip install nuitka
::::::  -standalone 独立可执行文件，包含所有依赖项。
::::::  -onefile 生成单个可执行文件，而不是多个文件。
::::::  -windows-icon-from-ico=icon.ico 指定窗口图标文件。
::::::  -windows-disable-console 禁用控制台窗口。
::::::  --inclde-package-date=PACKAGE 指定包中的数据文件
::::::  --include-data-dir=DIR 指定包含数据文件的目录。
::::::  --include-data-files=SRT=DEST 指定包含数据文件的源和目标路径。

::::::  nuitka --standalone --onefile --windows-icon-from-ico=icon.ico --windows-disable-console your_script.py
