import argparse
from typing import List
import itertools


def getData(data: List[str], min: int, max: int):
    max = max+1
    for i in range(min, max):
        result = list(itertools.combinations(data, i))
        result = [''.join(t) for t in result]
        # print(*result)
        yield result


def main(filePath: str, min: int, max: int, output: str):

    data: List[str] = None
    with open(filePath, "r", encoding="utf-8") as file:
        data = file.read() .splitlines()  # file.readlines()

    with open(output, "w", encoding="utf-8") as file:
        for arr in getData(data, min, max):
            r = [a+"\n" for a in arr]
            file.writelines(r)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成字典")
    group = parser.add_argument_group("获取配置")
    parser.add_argument("-f", "--file",  type=str, help="字典模板文件,按行分割")
    parser.add_argument("-m", '--min', type=int, help="最小个数,default:2", default=2)
    parser.add_argument("-x", '--max', type=int,  help="最大个数,default:5", default=5)
    parser.add_argument("-o", "--output",  type=str, help="输出文件,defautl:output.dict", default="output.dict")

    args = parser.parse_args()
    if args.file == None:
        parser.print_help()
    else:
        main(args.file, args.min, args.max, args.output)
