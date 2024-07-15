import os
import sys
from exec_command import execute_command
import argparse
import shutil
from co6co.utils import find_files, getWorkDirectory, get_parent_dir


def getApplicationDir():
    # 获取当前文件的完整路径
    script_path = __file__
    print("currentFilepath:", script_path)
    # 获取当前文件所在目录
    script_dir = os.path.dirname(script_path)
    print("current directory:", script_dir)

    print("如果脚本被编译成.pyc文件运行或者使用了一些打包工具（如PyInstaller），那么__file__可能不会返回源.py文件的路径，而是编译后的文件或临时文件的路径")
    # 获取当前文件的完整路径
    if getattr(sys, 'frozen', False):
        # 如果应用程序是冻结的，获取可执行文件的路径
        application_path = os.path.dirname(sys.executable)
    else:
        # 否则，获取原始脚本的路径
        application_path = os.path.dirname(os.path.abspath(__file__))

    print("Application path:", application_path)
    return application_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="audit service.")
    parser.add_argument('-g', '--gen', type=bool,
                        action=argparse.BooleanOptionalAction, help="打包")
    parser.add_argument('-d', '--delete', type=bool,
                        action=argparse.BooleanOptionalAction, help="删除临时目录和文件")
    parser.add_argument('-t', '--target',  help="复制.gz.tar 到指定目录")
    args = parser.parse_args()
    workDir = getWorkDirectory()
    project_dir = get_parent_dir(__file__, 3)

    print("当前工作目录：", workDir, "\n项目目录：", project_dir)
    # 生成打包
    if args.gen:
        fileName = "setup.py"
        gen = find_files(project_dir, None, lambda x: x ==
                         fileName, '.git', 'node_modules', 'src')
        for root, dirs, files in gen:
            # 没有符合条件的文件
            if len(files) == 0:
                continue
            os.chdir(root)
            print("切换工作目录：", getWorkDirectory())
            '''
            想读出版本信息和包名未成功
            if os.getcwd() not in sys.path:
                sys.path.append(os.getcwd())
            srcdirs=[ os.path.join(root, d) for d in dirs if "__pycache__" != d ]
            print(srcdirs) 
            sys.path.extend(srcdirs)
            from setup import VERSION
            print("斑斑",VERSION) 
            '''
            # shutil.copytree(root, os.path.join(workDir),dirs_exist_ok=True)
            execute_command(f"python setup.py sdist")

    # 删除***.egg-info临时文件夹
    if args.delete:
        gen = find_files(workDir, lambda f: "egg-info" in f,
                         None, '.git', 'node_modules', 'src')
        for root, dirs, files in gen:
            for d in dirs:
                dir = os.path.join(root, d)
                # 谨慎操作
                if os.path.exists(dir) and os.path.isdir(dir):
                    print("删除目录：", os.path.join(root, d))
                    # shutil.rmtree(dir) # 谨慎操作

    if args.target:
        extension = ".tar.gz"
        gen = find_files(project_dir, None,
                         lambda f: f.endswith(extension), '.git', 'node_modules', 'src')
        if not os.path.exists(args.target):os.makedirs(args.target)
        for root, dirs, files in gen:
            for file in files:
                # 不仅复制源文件的内容,尝试复制文件的元数据（如权限和时间戳）
                shutil.copy2(os.path.join(root, file), args.target)
                print(os.path.join(root, file), "--copy2-->", args.target)
