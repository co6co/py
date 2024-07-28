import os
from co6co.utils import log
import requests
import time
fileName='C:\\Users\\Administrator\\Desktop\\整理\\smoke.txt'

 
allData=[]
try:
    with open(fileName, 'r',encoding='utf-8') as file:
        lines = file.readlines()  
        boat=None
        for line in lines: 
            line=line.rstrip() 
            print(line)
            if line[0:1]=='[':
                if boat!=None:allData.append(boat)
                boat={} 
                boat['boat']=line
                boat['name']=line.split(']')[1].strip()
                boat['other']=[]
            else:
                boat['other'].append(line)
        #输出最好一个
        allData.append(boat)       
except FileNotFoundError:
    print(f"文件 {fileName} 不存在")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    log.err("error",e)
print(allData[9])



def query(name,f=0):
    json={"name":name ,"pageIndex":1,"pageSize":10,"treeFlag":f}
    headers={
        "Content-Typ":"application/json;charset=UTF-8",
        "Authorization":"Bearer xx"
    }
    res=requests.post("http://xx/audit3/v1/api/biz/group/list",headers=headers, json=json)
    data=res.json()['data']
    print(data)
    return data

print("查看从文件获得的数据：",allData[0])
for item in allData:
    # 父类
    name=item['name']
    data=query(name,0)
    if len(data)==1: 
        a=data[0]
        item['gs']=a['name'] 
        item['sn']=a['children'][0]["boatSerial"]
        
    # 子类
    data=query(name,1)
    item['cams']=[]
    for a in data:
        item['cams'].append({"name":a['name'],'sn':a['ipCameraSerial']})  
    time.sleep(0.1)


fileName=fileName+"_2.txt"
try:
    with open(fileName, 'w') as file:
        lines=[]
        for item in allData:
            lines.append(item["gs"])
            lines.append(f"{item['boat']}\t{item['sn']}")
            for other in item['other']:
                lines.append(other)
            if "cams" in item:
                for cam in item['cams']:
                    lines.append(f"\t{cam['name']}\t{cam['sn']}")
        newLine=[l+"\n" for l in line] 
        file.writelines(lines) 
        file.write('\n')  
except Exception as e:
    print(f"写入文件{fileName}时发生错误: {e}")