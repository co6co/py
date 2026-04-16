import  os,sys
import argparse

def main(args): 
    with open(args.source, 'r', encoding=args.encoding) as f:
        lines =f.readlines()
    with open(args.output, 'w',encoding=args.output_encoding) as f:
        f.writelines(lines)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Text Tool")
    parser.add_argument('-s', "--source", type=str, help="Input file path")
    parser.add_argument('-e', "--encoding", type=str, default="utf-8", help="源文件编码")
    parser.add_argument('-o', "--output", type=str,   default="./output/output.txt", help="Output file path")
    parser.add_argument('-oe', "--output_encoding", type=str, default="utf-8", help="输出文件编码")
    args = parser.parse_args() 
    if not args.source:
        parser.print_help()
        sys.exit(1) 
    if not os.path.exists(args.source):
        parser.error("Input file not found")
    dir = os.path.dirname(args.output)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    main(args)