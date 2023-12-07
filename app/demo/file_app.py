import os
import argparse 

from co6co.utils import hash  
 
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
            h=hash.file_md5(p)
            if repeat: hash_list.append(fileOption(filePath=p,hash=h)) 
            else: 
                h_l=[ h.hash for h in hash_list]  
                if h not in h_l:hash_list.append(fileOption(filePath=p,hash=h))  
    return hash_list

def main():
    parser=argparse.ArgumentParser(usage="查询MD5")
    parser.add_argument("-d","--folder",default=R"C:\Users\Administrator\Desktop\aw\recovery")
    parser.add_argument("-r","--repeat", help="dispaly repeat", action=argparse.BooleanOptionalAction, default=False)
    
    args=parser.parse_args()
    hash_list=get_hash_file(args.folder,args.repeat)
    print(args.repeat)
    for h in hash_list:
        print(h.filePath,'->',h.hash)

        
if __name__ =="__main__":
    main()
    