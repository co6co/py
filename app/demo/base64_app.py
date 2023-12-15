# -*- coding: utf-8 -*-
"""
对文件内容 base64 加解密
对文本内容 base64 加解密
"""
import argparse

from co6co.utils.File import File
from co6co.utils import hash


def main(args):
    try:
        if args.source==None:
            parser.print_help()
            exit()
        if args.isFile: 
            if args.target==None:
                print("有 -f 或者 --isFile,--target 必须存在")
                exit()
            if args.decode: 
                data=File.readFileBase642Str(args.source) 
                with open(args.target, "w") as f:
                    f.write(data)  
            else:
                data=File.readFile2Base64(args.source) 
                with open(args.target, "w") as f:
                    f.write(data)  
        else: 
            if args.decode:print(hash.debase64(args.source) )
            else:print( hash.enbase64(args.source))
    except Exception as e:
        print("输入参数可能不符合base64 规范,请检查输入参数：",e)
      
        
if __name__ =="__main__":
    parser=argparse.ArgumentParser(usage="base64 en/decode")
    parser.add_argument("-s","--source",type=str,required=True)
    parser.add_argument("-t","--target",type=str,help="--isFile 存在才有效")
    parser.add_argument("-f","--isFile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-d","--decode", action=argparse.BooleanOptionalAction, default=False)
    
    args=parser.parse_args()
    main(args)
    