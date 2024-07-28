import os
from co6co.utils import log
import requests
import time
fileName='C:\\Users\\Administrator\\Desktop\\整理\\boat.txt'

 
allData=[]
def open5( func):
    try:
        with open(fileName, 'r',encoding='utf-8') as file:
            lines = file.readlines()  
            boat=None
            for line in lines: 
                func(line)
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

def read1():
    try:
        with open(fileName, 'r',encoding='utf-8') as file:
            lines = file.readlines()   
            for line in lines:  
                line=line.strip()  
                allData.append({"boat":line,"name":line,'other':[]})    
    except FileNotFoundError:
        print(f"文件 {fileName} 不存在")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        log.err("error",e)  
read1()
print(allData[2])

 
def query(name,f=0):
    json={"name":name ,"pageIndex":1,"pageSize":10,"treeFlag":f}
    headers={
        "Content-Typ":"application/json;charset=UTF-8",
        "Authorization":"Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MjIyNTUyNzMsImlhdCI6MTcyMjE2ODg3MywiaXNzIjoiSldUK1NFUlZJQ0UiLCJkYXRhIjp7ImlkIjoxLCJ1c2VyTmFtZSI6ImFkbWluIiwiZ3JvdXBfaWQiOjF9fQ.dYzx14emuoxZAOn2h4OwRp-CAINcPVY93OmcGEAEmpw"
    }
    res=requests.post("http://audit-web.ngrok.jshwx.com.cn:28081/audit3/v1/api/biz/group/list",headers=headers, json=json)
    data=res.json()['data']
    #print(data)
    return data

#print("查看从文件获得的数据：",allData)
 
for item in allData:
    # 父类
    name=item['name']
    print("query",name)
    data=query(name,0)
    print(data)
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
            print("GG",item)
            print("GG",item["gs"])
            lines.append(item["gs"])
            lines.append(f"{item['boat']}\t{item['sn']}")
            for other in item['other']:
                lines.append(other)
            if "cams" in item:
                for cam in item['cams']:
                    lines.append(f"\t{cam['name']}\t{cam['sn']}")
        [file.write(l+"\n") for l in lines]   
        file.write('\n')  
except Exception as e:
    print(f"写入文件{fileName}时发生错误: {e}")
    log.err("dd",e)