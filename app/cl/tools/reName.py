import pathlib
from co6co.utils import find_files
dir=input("输入文件夹:")

def getEnableNewName(filePath:pathlib.Path):
    suffix=filePath.suffix
    name=filePath.stem
    flag=0
    while filePath.exists():
        flag+=1 
        fileName=f"{name}_{flag}{suffix}"
        filePath=filePath.parent / fileName
        #pathlib.Path(newName).rename(f"{name}_{flag}{suffix}")
    return filePath

 
def rename(name,newName):
    if pathlib.Path(newName).exists():
        newName

if pathlib.Path(dir).exists():
    g=find_files(pathlib.Path(dir))
    for root,s1,f in g: 
        for f1 in f:
            index=f1.find("#")
            name=f1[0:index]
            pathlib.Path(f1).suffix
            newPath=pathlib.Path(root) / f"{name}{pathlib.Path(f1).suffix}"  
            print(f1,"->\t",newPath)
            newPath=getEnableNewName(newPath)
            oldPath:pathlib.Path=(pathlib.Path(root) / f1)
            oldPath.rename(newPath)
            print(newPath)
            #print(newName)


