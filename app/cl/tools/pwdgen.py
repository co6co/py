 



def getData(p,m,a):
    for p1 in p:
        for m1 in m:
            for a1 in a:
                yield p1+m1+a1
def cc(a1,a2,a3):
    data=[]
    for item in getData(a1,a2,a3):
        #if item.startswith("@") or item.startswith("_"):
        #    continue 
        data.append(item)
    return data
file="./dist/pwdgen.txt"
with open(file,"w+") as f:
    for item in data:
        f.write(item+"\n")

print("生成密码总数:", len(data))


