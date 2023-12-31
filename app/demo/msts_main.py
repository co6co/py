
import co6co.utils.http as http
import co6co.utils.log as log 
import socket,socks,threading
import re,requests,os,sys
from urllib.parse import urlparse,unquote_plus,quote_plus,parse_qsl
import concurrent.futures as futures
from bs4 import BeautifulSoup

from multiprocessing import cpu_count
import argparse,time

class msts:
    @staticmethod
    def set_proxy(ip:str, port:int):
        socks.set_default_proxy(socks.SOCKS5, ip, port)
        socket.socket = socks.socksocket

    def __init__(self,listUrl:str,workers_count:int=1,worker_sleep_sec:int=3, downloadDir="D:\\temp") -> None:
        self.url=listUrl 
        self.downloadFolder=downloadDir
        self.worker_count=workers_count
        self.worker_sleep_sec=worker_sleep_sec
        self.__thread_lock__=threading.RLock()
        self.__referer_address__=None
        self.__mhost_address__=None
        if not os.path.exists(self.downloadFolder):os.makedirs(self.downloadFolder)
        result=urlparse(listUrl)
        self.host=f"{result.scheme}://{result.netloc}"
        pass
    def _get(self,url)->str:
        headers={"User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36"}
        response=http.get(url,header_dict=headers,verify=False)
        response.encoding="utf-8"
        return response.text
    def _getList(self)->list|None:
        text=self._get(self.url)
        f=r"/play_m/\d+_\d+_\d+_\d+.html"
        kk = re.compile(f)
        m=re.findall(kk,text)
        return m
    def _getMhost(self,url:str,referer:str)->str|None:
        try:
            self.__thread_lock__.acquire()
            if self.__mhost_address__ !=None:
                return self.__mhost_address__
            headers={
                "User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36",
                "Referer": referer, 
            }
            response=http.get(url,header_dict=headers,verify=False)
            resource=re.search('var\s+u[\s]{0,}=[\s]{0,}\"\*.*',response.text)
            if resource!=None:
                #资源主机
                mhost=""
                chars=resource.group(0).split("*")
                for i in chars[1:-1]:
                    mhost+=f"{chr(int(i))}"
                self.__mhost_address__=mhost
                return mhost
        finally:
            self.__thread_lock__.release()
        
    
    def _getReferer(self,url:str)->str|None:
        try:
            result=urlparse(url)
            arr=re.findall("\d+",result.path)
            id=arr[0]
            said=arr[1]
            num=arr[3]
            api=f"https://ai-m-5tps.iiszg.com/play-m.php?url=XXXXXXXX&jiidx=/play_m/{id}_{num}_1_{int(num)+1}.html&jiids=/play_m/{id}_{num}_1_{int(num)-1}.html&id={id}&ji={num}&said={said}"
            self.__thread_lock__.acquire()
            if self.__referer_address__ !=None:
                r=self.__referer_address__
                
                data=parse_qsl(urlparse(unquote_plus(r)).query)[0][1]
                #第一次请求时的文件名，不需要后缀
                oldName=os.path.basename(data)[0:-4]
                text=c._get(url)
                bt=BeautifulSoup(text,"html.parser")
                h3=bt.find("h3")
                newFileName=h3.getText().split(":")[1]
                data=data.replace(oldName,newFileName) 
                
                '''
                #采用更改数字方式修改文件名
                regex = re.compile(r"\d{1,}") 
                numData=re.search(regex,data).group(0)
                data=regex.sub(f'{int(num):0{len( numData)}d}',data,1)
                '''
                r=api.replace("XXXXXXXX",requests.utils.requote_uri(data))
                return r
            text=c._get(url)
            f='var\splay_link[\s]{0,}=[\s]{0,}\".*'
            m=re.search(f,text)
            if m!=None:
                r=m.group(0)
                index=r.index('"')
                r=r[index+1:]
                index=r.index('"')
                r=r[0:index]
                
                r=r.replace("&id=&ji=",f"&id={arr[0]}&ji=")
                self.__referer_address__=r
                return r
        finally:
            self.__thread_lock__.release()


    def _parser(self,url)->(str,str):
        """
        url: 播放页 url
        Referer,mp3
        """ 
        r=self._getReferer(url)
        if r==None: 
            log.err(f"请求‘{url}’未能获得Referer 地址.")
            return (None,None) 
            
        mhost=self._getMhost(r,url)
        if mhost==None: 
            log.err(f"请求‘{url}’未能获得mHost 地址.")
            return (None,None)
        
        deUrl=unquote_plus(r,encoding="utf-8") 
        data=parse_qsl(deUrl)
        mp3Url=data[0][1]  
        return (r,mhost+mp3Url)
    
    def _download(self,itemUrl:str,error_times:int=0): 
        try:
            if error_times==3:
                log.warn(f"{itemUrl}重试{error_times}次未能下载成功")
                return
            log.start_mark(f"开始下载...{itemUrl}")
            r,m=self._parser(itemUrl)
            log.warn(f"{r}\r\n{m}")
            fileName=os.path.basename(m)

            #header={"Accept": ""*/*","Referer":r,"Sec-Ch-Ua-Platform":"Android","Sec-Ch-Ua":'"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',"Sec-Fetch-Dest":"audio","User-Agent":"Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.109 Safari/537.36 CrKey/1.54.248666"}
            header={
                "Referer": r, 
                "User-Agent": 'Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.109 Safari/537.36 CrKey/1.54.248666'
            } 
            status,data=http.download(requests.utils.requote_uri(m),timeout=30,header_dict=header,verify=False)
            if status == 200 or status == 206:
                log.warn(os.path.join(self.downloadFolder,fileName))
                File=open(os.path.join(self.downloadFolder,fileName),"wb") # Opens the file for writing.
                File.write(data)
                log.end_mark(f"下载{itemUrl}完成.")
            else:
                log.end_mark(f"下载{itemUrl}出错.{status}")
                self._download(itemUrl,error_times+1)
            time.sleep(self.worker_sleep_sec)
        except Exception as e:
            log.err(f"下载“{itemUrl}error:{e}")
            self._download(itemUrl,error_times+1)
            

    def downloads(self):
        log.start_mark("解析List...")
        list=self._getList()
        log.end_mark("解析List.")
        if list !=None:
            log.end_mark(f"解析List.{len(list)}")
            with futures.ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                futures_dict= {executor.submit(self._download,f"{self.host}{itemUrl}"):itemUrl for itemUrl in list } 
                futures.wait(futures_dict)


if __name__ == '__main__' :
    default_save_dir=os.path.join(os.path.abspath("."),"Download") 
    parser=argparse.ArgumentParser(description="下载文件")
    parser.add_argument("-u","--url",type=str, help="需要下载的URL 列表页")
    
    parser.add_argument("-p","--proxy",type=str, help="代理服务器,ip:port",default=None)
    parser.add_argument("-s","--sleep",type=int, help="sleep secounds",default=3)
    parser.add_argument("-w","--works",type=int, help="sleep secounds",default=cpu_count())
    parser.add_argument('--list' ,default=True, action=argparse.BooleanOptionalAction,help=f"Url 是列表页")
    parser.add_argument("-d","--folder",type=str,help=f"保存目录 [{default_save_dir}]",default=default_save_dir)
    args=parser.parse_args()
    c=msts(args.url,worker_sleep_sec=args.sleep, downloadDir=args.folder,workers_count=args.works)
    if args.proxy !=None:
        arr = args.proxy.split(":")
        msts.set_proxy(arr[0],int(arr[1]))
    if args.url == None:
        parser.print_help()
        sys.exit(0)
    if args.list: c.downloads()
    else: c._download(args.url)

