import os
import argparse


def delete_files(folder: str, textFilePath: str, isAbs: bool):
    """
    删除指定文件夹中的文件，文件列表从文本文件中读取。

    :param folder: 文件夹路径，如果文本文件中的文件名不是绝对路径，则使用此文件夹作为基础路径。
    :param textFilePath: 包含文件名列表的文本文件路径。
    :param isAbs: 指示文本文件中的文件名是否为绝对路径。
    """
    # 检查文件列表是否存在
    if not os.path.exists(textFilePath):
        print(f"Error: 文件列表 {textFilePath} 不存在！")
        return

    # 读取文件名列表
    with open(textFilePath, 'r', encoding='utf-8') as file:
        # 按行读取并去除换行符
        filenames = file.read().splitlines()

    # 遍历文件名列表并删除文件
    for filename in filenames:
        # 去除多余的空格
        filename = filename.strip()
        # 跳过空行
        if not filename:
            continue

        if isAbs:
            # 直接使用文件名作为路径
            file_path = filename
        else:
            # 构建文件路径
            file_path = os.path.join(folder, filename)
        # 获取绝对路径
        file_path = os.path.abspath(file_path)
        try:
            # 检查文件是否存在
            if os.path.exists(file_path):
                # 删除文件
                os.remove(file_path)
                print(f"成功删除文件: {file_path}")
            else:
                print(f"警告: 文件 {file_path} 不存在，跳过删除。")
        except Exception as e:
            print(f"错误: 删除文件 {file_path} 失败，原因: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除指定文件夹中的文件")
    delFileList = "delFileList.txt"
    parser.add_argument("-f", "--folder",  type=str, help=f"文件夹")
    parser.add_argument("-t", "--textPath", default=delFileList, type=str, help=f"数据库所在文件夹default:{delFileList}")
    parser.add_argument("-a", "--abs", default=False, action=argparse.BooleanOptionalAction, type=bool, help=f"文本内容是否是绝对路径default:False")
    args = parser.parse_args()
    if args.folder and not args.abs:
        delete_files(args.folder, args.textPath, args.abs)
    elif args.textPath and args.abs:
        delete_files(None, args.textPath, args.abs)
    else:
        parser.print_help()
