 
import argparse



def jointGenertor(p,m,a):
    for p1 in p:
        for m1 in m:
            for a1 in a: 
                yield p1+m1+a1
def getData(prefix,middle,append):
    print("使用字符串：",locals())
    data=[]
    for item in jointGenertor(prefix,middle,append):
        #if item.startswith("@") or item.startswith("_"):
        #    continue 
        data.append(item)
    return data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成密码")
    parser.add_argument("-p", "--prefix",  default="admin,abc", type=str, help="密码前缀:admin,abc")
    parser.add_argument("-m", "--middle",  default="_,@,", type=str, help="密码中间部分:_,@,")
    parser.add_argument("-a", "--append",  default="123,123456,12345678", type=str, help="密码后缀")
    parser.add_argument("--append2",  default="", type=str, help="密码后缀2")
    parser.add_argument("-o", "--output",  default="./dist/pwdgen.txt", type=str, help="输出文件路径")
    args = parser.parse_args()
    if not args.prefix and not args.middle and not args.append:
        parser.print_help()
        exit(1)
    file = args.output
    prefix = args.prefix.split(",")
    middle = args.middle.split(",")
    append = args.append.split(",")
    append2 = args.append2.split(",")
    data2=getData(prefix,middle,append)
    
    if len(append2)>0 and append2[0]!="":
        data=getData(data2,append2,[''])
    else:
        data=data2
    with open(file,"w+") as f:
        for item in data:
            f.write(item+"\n")
    print("生成密码总数:", len(data))


 

