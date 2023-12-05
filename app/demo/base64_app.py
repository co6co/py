import base64 
import os
import argparse
  
'''data = "Python is a programming language,随" 
data_bytes = data.encode('utf-8')  
data=base64.b64encode(data_bytes) 
data=base64.b64decode(data) 
'''
 
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
        if args.decode: base64.decode(open(args.source), open(args.target, "w"))
        else:base64.encode(open(args.source), open(args.target, "w"))
    else:
        data_bytes = args.source.encode('utf-8')  
        if args.decode:print(base64.b64encode(data_bytes) )
        else:print( base64.b64encode(data_bytes).decode("utf-8") )
      
        
if __name__ =="__main__":
    main()
    