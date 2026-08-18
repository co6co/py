echo 开始打包...
echo 使用到 deppends.exe,winlibs-x86_65-posix-seh-gcc,mingw-w64msvcrt

python -m nuitka --standalone --onefile --windows-disable-console --enable-plugin=pylint-warnings .\main.py