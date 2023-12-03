import os
import argparse
import hashlib


def file_hash(file_path: str, hash_method) -> str:
    if not os.path.isfile(file_path):
        print('文件不存在。')
        return ''
    h = hash_method()
    with open(file_path, 'rb') as f:
        while b := f.read(8192):
            h.update(b)
    return h.hexdigest() 

def str_hash(content: str, hash_method, encoding: str = 'UTF-8') -> str:
    return hash_method(content.encode(encoding)).hexdigest()


def file_md5(file_path: str) -> str:
    return file_hash(file_path, hashlib.md5)


def file_sha256(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha256)


def file_sha512(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha512)


def file_sha384(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha384)


def file_sha1(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha1)


def file_sha224(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha224)


def str_md5(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.md5, encoding)


def str_sha256(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha256, encoding)


def str_sha512(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha512, encoding)


def str_sha384(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha384, encoding)


def str_sha1(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha1, encoding)


def str_sha224(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha224, encoding) 
 
class fileOption:
    filePath:str
    hash:str
    def __new__(cls,**kvargs) :
        obj=object.__new__(cls)
        obj.__dict__.update(kvargs)
        return  obj     

def get_hash_file(folder:str,repeat:bool): 
    hash_list:list[fileOption]=[] 
    #file=os.listdir(folder) 
    for parent, dirnames, filenames in os.walk(folder):
        # Case1: traversal the directories
        for dirname in dirnames:
            print("Parent folder:", parent)
            print("Dirname:", dirname)
        # Case2: traversal the files
        for filename in filenames:
            print("Parent folder:", parent)
            print("Filename:", filename) 
            p=os.path.join(parent,filename)
            hash=file_md5(p)
            if repeat: hash_list.append(fileOption(filePath=p,hash=hash)) 
            else: 
                h_l=[ h.hash for h in hash_list]  
                if hash not in h_l:hash_list.append(fileOption(filePath=p,hash=hash))  
    return hash_list

def main():
    parser=argparse.ArgumentParser(usage="查询MD5")
    parser.add_argument("-d","--folder",default=R"C:\Users\Administrator\Desktop\aw\recovery")
    parser.add_argument("-c","--repeat", action=argparse.BooleanOptionalAction, default=False)
    
    args=parser.parse_args()
    hash_list=get_hash_file(args.folder,args.repeat)
     
    for h in hash_list:
        print(h.filePath,'->',h.hash)

        
if __name__ =="__main__":
    main()
    