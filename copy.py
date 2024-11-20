

import os
import shutil
import argparse
def compare_versions(v1:str, v2:str):
    """
    return v1>v2=1,v1<v2=-1,v1<v2=0
    """
    # 将版本号按点号分割成列表，并将每个元素转换为整数
    v1_parts = list(map(int, v1.split('.')))
    v2_parts = list(map(int, v2.split('.')))

    # 使用zip函数对两个列表进行比较
    for part1, part2 in zip(v1_parts, v2_parts):
        if part1 > part2:
            return 1
        elif part1 < part2:
            return -1
    
    # 如果长度不同，且前面的部分都相同，则更长的那个版本号更大
    if len(v1_parts) > len(v2_parts):
        return 1
    elif len(v1_parts) < len(v2_parts):
        return -1
    else:
        return 0
def main(source:str, targetFolder:str):
    from co6co.utils import log
    from co6co.utils import find_files  
    import re
    folderFilter=lambda d:d=='dist'
    fileFilter=lambda f: ".tar.gz" in f
    ignoreFolder=['.git','__pycache__','node_modules']
    generator=find_files(source,*ignoreFolder,filterDirFunction=folderFilter ,filterFileFunction=fileFilter )
    red=r'\d+(.\d+){2,3}' 
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    for root,_,files in generator: 
        if  len(files)==0:continue 
        maxFilePath:str=None
        maxVersion:str=None
        for file in files:  
            match=re.search(red,file) 
            if match==None:
                print("{}不能匹配版本号{}".format(file,red))
                continue 
            version:str=match.group()
            s=os.path.join(root,file)   
            if maxVersion==None:
                maxVersion=version
                maxFilePath=s
            else:
                v=compare_versions(maxVersion,version) 
                if v==-1:
                    maxVersion=version
                    maxFilePath=s
        print("复制文件：{}->{}".format(maxFilePath,targetFolder))
        shutil.copy2(maxFilePath, targetFolder) 
if __name__ =="__main__": 
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="复制py包") 
    parser.add_argument("-s", "--source",  type=str, default='.' ,help="源目录")
    parser.add_argument("-t", "--target",  type=str, default='../build' ,help="目标目录")
    args = parser.parse_args()
    if args.target == None:
        parser.print_help()
    else: 
        main(args.source, args.target)
