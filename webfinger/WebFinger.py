#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: WebFinger.py
@Time: 2019/11/08 20:26
@About: 
'''

from socket import *
import requests
import lxml,random
from bs4 import BeautifulSoup
import sys,os,re
import sqlite3
import concurrent.futures
from typing import Tuple,List
import argparse


requests.packages.urllib3.disable_warnings()
class WebFinger():
    dbFile:str="./cms_finger.db"
    def __init__(self, host, threadNum=30):
        self.host =   host
        self.finger = []
        self.re_title = re.compile(r'title="(.*)"')
        self.re_header = re.compile(r'header="(.*)"')
        self.re_body = re.compile(r'body="(.*)"')
        self.re_bracket = re.compile(r'\((.*)\)')
        self.threads = threadNum

        

    def run(self):
        print("-"*20 + "Start WebFinger Matching" + "-"*20)
        if self.thread():
            print("[+] " + self.host +" use:")
            result = ""
            for i in self.finger:
                result += i + "  "
            print(self.finger)
            print("[+] fofa_banner: " + result)
            print("-"*22 + "End WebFinger Matching" + "-"*20)

    def get_data(self)->Tuple[str,dict,str]: 
        '''
        获取web 的 title,header,body
        '''
        data = requests.get(self.host, headers=self.set_header(), timeout=3, verify=False)
        content = data.text
        title = BeautifulSoup(content, "lxml").title.text.strip()
        return title.strip('\n'),data.headers, content, 

    def set_header(self):
        user_agent = ['Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.8.1) Gecko/20061010 Firefox/2.0',
                    'Mozilla/5.0 (Windows; U; Windows NT 5.0; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.6 Safari/532.0',
                    'Mozilla/5.0 (Windows; U; Windows NT 5.1 ; x64; en-US; rv:1.9.1b2pre) Gecko/20081026 Firefox/3.1b2pre',
                    'Opera/10.60 (Windows NT 5.1; U; zh-cn) Presto/2.6.30 Version/10.60','Opera/8.01 (J2ME/MIDP; Opera Mini/2.0.4062; en; U; ssr)',
                    'Mozilla/5.0 (Windows; U; Windows NT 5.1; ; rv:1.9.0.14) Gecko/2009082707 Firefox/3.0.14',
                    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
                    'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr; rv:1.9.2.4) Gecko/20100523 Firefox/3.6.4 ( .NET CLR 3.5.30729)',
                    'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr-FR) AppleWebKit/528.16 (KHTML, like Gecko) Version/4.0 Safari/528.16',
                    'Mozilla/5.0 (Windows; U; Windows NT 6.0; fr-FR) AppleWebKit/533.18.1 (KHTML, like Gecko) Version/5.0.2 Safari/533.18.5']
        UA = random.choice(user_agent)
        headers = {
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'User-Agent':UA,
        'Upgrade-Insecure-Requests':'1','Connection':'keep-alive','Cache-Control':'max-age=0',
        'Accept-Encoding':'gzip, deflate, sdch','Accept-Language':'zh-CN,zh;q=0.8' 
        #"Referer": "https://www.baidu.com/link?url=Ni7wOsdwUuz50m1no12V0q3QtNYXbNgXoybY9SUqoKG",
        #'Cookie':"PHPSESSID=gljsd5c3ei5n813roo4878q203"
        }
        return headers
    
    def count(self)->int:
        '''
        获取表fofa总记录数 
        '''
        with sqlite3.connect(f"{os.getcwd()}{ self. dbFile}") as conn:
            cursor = conn.cursor()
            result = cursor.execute('select count(id) from `fofa`')
        for row in result:
            return row[0] 
 
    @classmethod 
    def select(clc, name:str)->List[Tuple[str,str]]|None:
        '''
        通过名称查询 name,keys
        ''' 
        with sqlite3.connect(f"{os.getcwd()}{ clc. dbFile}") as conn:
            cursor = conn.cursor()
            inject=['--','#','(',')',',','||','+','\'','&',';','%','@','"','\\\'','\\"',"<",">"]
            la=lambda c:c in inject
            for r in name:
                # 是否存在注入攻击
                if la(r):return None 
            result = cursor.execute(f'select name,keys from fofa where name like \'%{name}%\'') 
            print(type(result))
            li=[]
            for row in result: 
                li.append((row[0], row[1]))
            return li
    def selectByIndex(self, x:int)->Tuple[str,str]:
        '''
        获取第 x 条记录的 name,keys
        '''
        with sqlite3.connect(os.getcwd() + "./cms_finger.db") as conn:
            cursor = conn.cursor()
            result = cursor.execute(f'select name,keys from fofa limit {x-1},1') 
            for row in result: return row[0], row[1]    
    
    def check(self, key, header:str, body:str, title:str)->bool:
        '''
        key 的内容是否在  title body header
        param:
        '''
        if 'title="' in key:
            if re.findall(self.re_title, key)[0].lower() in title.lower():
                return True
        elif 'body="' in key:
            if re.findall(self.re_body, key)[0] in body: 
                return True
        else:
            if re.findall(self.re_header, key)[0] in header: 
                return True

    def match(self, x, header, body, title):
        name, key = self.selectByIndex(x)
        if '(' not in key:
            if '&&' not in key:
                if '||' not in key:
                    if self.check(key, header, body, title):
                        self.finger.append(name)
                elif '||' in key:
                    for s in key.split('||'):
                        if self.check(s, header, body, title):
                            self.finger.append(name)
                            break
            elif '&&' in key and '||' not in key:
                times = 0
                for s in key.split('&&'):
                    if self.check(s, header, body, title):
                        times += 1
                if times == len(key.split('&&')):
                    self.finger.append(name)
        else:
            if '&&' in s.findall(self.re_bracket, key)[0]:
                for s in key.split('||'):
                    if '&&' in s:
                        times = 0
                        for _re in key.split('&&'):
                            if self.check(_re, header, body, title):
                                times += 1
                        if times == len(key.split('&&')):
                            self.finger.append(name)
                            break
                    else:
                        if self.check(s, header, body, title):
                            self.finger.append(name)
                            break
            else:
                for s in key.split('&&'):
                    times = 0
                    if '||' in s:
                        for _re in key.split('||'):
                            if self.check(_re, title, body, header):
                                times += 1
                                break
                    else:
                        if self.check(s, title, body, header):
                            times += 1
                if times == len(key.split('&&')):
                    self.finger.append(name)
        
    def thread(self) -> bool:
        title,header, body  = self.get_data()  
        executor = concurrent.futures.ThreadPoolExecutor(max_workers = self.threads)
        futures = (executor.submit(self.match, sql, header, body, title) for sql in range(0, int(self.count())))
        concurrent.futures.wait(futures) 
        return True

            
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="检测网站指纹")

    parser.add_argument('--checkurl', default=True, action=argparse.BooleanOptionalAction ,help="check network connect and check port ,default=true")
    group=parser.add_argument_group("url check") 
    group.add_argument("-u","--url",type=str, help="输入要检测的网址")
    group.add_argument("-t","--threadcount",help="线程数" ,type= int,default=10)
    group=parser.add_argument_group("key select")
    group.add_argument("-n","--name",type=str, help="name")
   
    args=parser.parse_args()
    print(args)
    if args.checkurl and ( args.url==None or args.url==""):
        parser.print_help()
        sys.exit()
    elif args.checkurl:
        web = WebFinger(args.url,args.threadcount)
        web.run()
    else:  
        lst=WebFinger.select(args.name)
        for a,b in lst:
            print(a.rjust(50),"\t",b)
