import argparse
import json


def parseJsonFile(file: str, *arg, **kvargs):
    t = []
    with open(file, 'r') as file:
        # 将JSON数据加载到变量data中
        data: dict = json.load(file)
        for a in kvargs:
            if kvargs[a] != None and kvargs[a] in data:
                t.append(data.get(kvargs[a]))
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取Json文件")
    parser.add_argument("-f",  "--file", required=True, help="文件")
    parser.add_argument("-a", "--a", required=False,  help="参数1")
    parser.add_argument("-b", "--b",  required=False, help="参数2")
    parser.add_argument("-c", "--c",  required=False, help="参数3")
    parser.add_argument("-d", "--d",  required=False, help="参数4")
    parser.add_argument("-e", "--e",  required=False, help="参数5")

    args = parser.parse_args()
    if args.file == None:
        parser.print_help()
        exit(0)
    result = parseJsonFile(args.file, 123, 23, a=args.a,
                           b=args.b, c=args.c, d=args.d, e=args.e)
    print(*result)
