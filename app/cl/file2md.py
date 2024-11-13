"""
文件只是临时使用
为以前写的一些md文件做兼容性代码

"""
import os
from co6co.utils import log
from co6co.utils import find_files
from typing import Generator, List, Any

from pathlib import Path
from datetime import datetime
import shutil


def fileOpter(filePath, name, c, targetFilePath, subPath="/dev/route"):
    timestamp = os.path.getctime(filePath)
    dateTimeStr = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(filePath, dateTimeStr)
    # 先备份原文件内容
    with open(filePath, 'r', encoding='utf-8') as file:
        original_content = file.read()

    # 在开始位置插入新内容
    content_to_insert = f'''---
layout: post
title: {name}
subtitle:
date:       {dateTimeStr}
categories: [{c}]
tags: [{c},{name}]
---

'''
    # print(content_to_insert)
    original_content = original_content.replace("./img", '/static{}/img'.format(subPath))
    new_content = content_to_insert + '\n' + original_content
    # 将新的内容写回文件
    with open(targetFilePath, 'w', encoding='utf-8') as file:
        file.write(new_content)


def mdAppendData(docRoot: str):
    def fileFilter(fileName):
        md = ['.md']
        _, e = os.path.splitext(fileName)
        return e and str(e).lower() in md
    generator = find_files(docRoot, ".git", filterFileFunction=fileFilter)
    target = "H:\\Work\\Projects\\github\\doc\\_old"
    for folder, folderNameList, fileList in generator:
        for name in fileList:
            try:
                filePath = os.path.join(folder, name)
                subPath = folder.removeprefix(docRoot).replace("\\", "/")
                n, e = os.path.splitext(name)
                old = os.path.abspath(docRoot)
                tar = os.path.abspath(target)
                targetFilePath = filePath.replace(old, tar)
                if not os.path.exists(os.path.dirname(targetFilePath)):
                    os.makedirs(os.path.dirname(targetFilePath))
                fileOpter(filePath, n, os.path.basename(folder), targetFilePath, subPath)
            except Exception as e:
                log.err("eoor", name, folder, e)


def moveImage(src: str, target: str):
    def fileFilter(fileName):
        img = [".jpg", '.jpeg', '.png', '.webp', '.bmp']
        _, e = os.path.splitext(fileName)
        return e and str(e).lower() in img
    generator = find_files(src, ".git", filterFileFunction=fileFilter)
    for folder, folderNameList, fileList in generator:
        for name in fileList:
            try:
                filePath = os.path.join(folder, name)
                _, e = os.path.splitext(name)
                old = os.path.abspath(src)
                tar = os.path.abspath(target)
                targetFilePath = filePath.replace(old, tar)
                if not os.path.exists(os.path.dirname(targetFilePath)):
                    os.makedirs(os.path.dirname(targetFilePath))
                print("复制文件：", filePath, '-->', targetFilePath)

                shutil.copyfile(filePath, targetFilePath)
            except Exception as e:
                log.err("eoor", name, folder, e)


if __name__ == "__main__":
    mdAppendData('I:\\document\\doc-bck')
    # moveImage('I:\\document\\doc-bck', 'I:\\document\\doc-img')
