
import co6co.utils.http as http
import co6co.utils.log as log 
import socket,socks
import re,requests,os,sys
from urllib.parse import urlparse,unquote_plus,quote_plus,parse_qsl
import concurrent.futures as futures

from multiprocessing import cpu_count
import argparse

class msts:
    @staticmethod
    def set_proxy(ip:str, port:int):
        socks.set_default_proxy(socks.SOCKS5, ip, port)
        socket.socket = socks.socksocket

    def __init__(self,listUrl:str,mp3Url:str,downloadDir="D:\\temp") -> None:
        self.url=listUrl
        self.mp3host="https://177h.wodeshougong.com/" if mp3Url ==None else mp3Url
        self.downloadFolder=downloadDir
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
    def _parser(self,url)->(str,str):
        """
        url: 播放页 url
        Referer,mp3
        """
        text=c._get(url)
        f='var\splay_link[\s]{0,}=[\s]{0,}\".*'
        m=re.search(f,text)
        if m!=None:
            r=m.group(0)
            index=r.index('"')
            r=r[index+1:]
            index=r.index('"')
            r=r[0:index]
            
            result=urlparse(url)
            arr=re.findall("\d+",result.path)

            r=r.replace("&id=&ji=",f"&id={arr[0]}&ji=")
            deUrl=unquote_plus(r)
            data=parse_qsl(deUrl)
            mp3Url=data[0][1] 
            return (r,self.mp3host+mp3Url)
    
    def _download(self,itemUrl:str): 
        log.start_mark(f"开始下载...{itemUrl}")
        r,m=self._parser(itemUrl)
        log.warn(f"{r}\r\n{m}")
        fileName=os.path.basename(m)
        header={"Referer":r}
        
        log.info(r+"\r\n"+m)
        status,data=http.download(requests.utils.requote_uri(m),timeout=30,header_dict=header,verify=False)
        if status == 200:
            log.warn(os.path.join(self.downloadFolder,fileName))
            File=open(os.path.join(self.downloadFolder,fileName),"wb") # Opens the file for writing.
            File.write(data)
            log.end_mark("开始下载.")
        else:
            log.end_mark(f"下载出错.{status}")

    def downloads(self):
        log.start_mark("解析List...")
        list=self._getList()
        log.end_mark("解析List.")
        if list !=None:
            log.end_mark(f"解析List.{len(list)}")
            with futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures_dict= {executor.submit(self._download,f"{self.host}{itemUrl}"):itemUrl for itemUrl in list } 
                futures.wait(futures_dict)


if __name__ == '__main__' :
    default_save_dir=os.path.join(os.path.abspath("."),"Download") 
    parser=argparse.ArgumentParser(description="下载文件")
    parser.add_argument("-u","--url",type=str, help="需要下载的URL 列表页")
    parser.add_argument("-m","--murl",type=str, help="媒体主机",default=None)
    parser.add_argument("-p","--proxy",type=str, help="代理服务器，ip:port",default=None)
    parser.add_argument('--list' ,default=True, action=argparse.BooleanOptionalAction,help=f"Url 是列表页")
    parser.add_argument("-d","--folder",type=str,help=f"保存目录 [{default_save_dir}]",default=default_save_dir)
    args=parser.parse_args()
    c=msts(args.url,mp3Url=args.murl,downloadDir=args.folder)
    if args.proxy !=None:
        arr = args.proxy.spit(":")
        msts.set_proxy(arr[0],int(arr[1]))
    if args.url == None:
        parser.print_help()
        sys.exit(0)
    if args.list: c.downloads()
    else: c._download(args.url)

