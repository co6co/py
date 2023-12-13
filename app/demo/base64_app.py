import base64 
import os
import argparse

from co6co.utils.File import File
'''data = "Python is a programming language,随" 
data_bytes = data.encode('utf-8')  
data=base64.b64encode(data_bytes) 
data=base64.b64decode(data) 
'''
 

def encode_file_to_base64(filename):
    with open(filename, 'rb') as file:
        file_content = file.read()
 
    # base64字符集
    base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
 
    # 将文件内容按6位分组，并在末尾补0
    groups = [file_content[i:i+3] for i in range(0, len(file_content), 3)]
    encoded_groups = []
    for group in groups:
        # 获取每个字符的ASCII码，转为二进制
        binary_group = ''.join(format(byte, '08b') for byte in group)
 
        # 根据6位进行切片并加密
        encodings = [base64_chars[int(binary_group[i:i+6], 2)] for i in range(0, len(binary_group), 6)]
 
        # 处理补位（如果不是3字节的整数倍）
        if len(encodings) < 4:
            encodings.extend(['='] * (4 - len(encodings)))
 
        encoded_groups.append(''.join(encodings))
    encoded_string = ''.join(encoded_groups)
    
    return encoded_string
 
 

def main():
    parser=argparse.ArgumentParser(usage="base en/decode")
    parser.add_argument("-s","--source",type=str,required=True)
    parser.add_argument("-t","--target",type=str)
    parser.add_argument("-f","--isFile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-d","--decode", action=argparse.BooleanOptionalAction, default=True)
    
    args=parser.parse_args()
    if args.source==None:
        parser.print_help()
        exit()
    if args.isFile:
        print(args.source,args.target) 
        if args.decode:
            base64.decode(open(args.source,"rb").raw, open(args.target, "wb"))
        else:
            data=encode_file_to_base64(args.source) 
            with open(args.target, "w") as f:
                f.write(data) 
                #File.writeBase64ToFile(args.target+".ttt",data)
            #base64.encode(open(args.source), open(args.target, "w"))
    else:
        data_bytes = args.source.encode('utf-8')  
        if args.decode:print(base64.b64encode(data_bytes) )
        else:print( base64.b64encode(data_bytes).decode("utf-8") )
      
        
if __name__ =="__main__":
    main()
    