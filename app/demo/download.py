import co6co.utils.http as http
import co6co.utils.log as log
import requests,os,sys,datetime
import socket,socks

import argparse,time

import m3u8 
#from Crypto.Util.Padding import pad
#from Crypto.Cipher import AES
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad
import glob
from concurrent.futures import ThreadPoolExecutor

 
def set_proxy(ip:str,port:int)->None:
    socks.set_default_proxy(socks.SOCKS5, ip, port)
    socket.socket = socks.socksocket
        
 
def download_ts(url, key, iv, headers:dict,tempFolder:str):
    r = requests.get(url, headers=headers)
    data = r.content
    if key !=None:data = AESDecrypt(data, key=key, iv=iv)
    with open(f"{tempFolder}/{iv:0>5d}.ts", "ab") as f:
        f.write(data)
    print(f"\r{iv:0>5d}.ts已下载", end=" ")


def get_real_url(url, headers:dict)-> m3u8.M3U8| str:
    log.warn(url)
    playlist = m3u8.load(uri=url, headers=headers)
    if len(playlist.playlists)==0:return playlist 
    return playlist.playlists[0].absolute_uri


def AESDecrypt(cipher_text, key, iv):
    cipher_text = pad(data_to_pad=cipher_text, block_size=AES.block_size)
    aes = AES.new(key=key, mode=AES.MODE_CBC, iv=iv)
    cipher_text = aes.decrypt(cipher_text)
    return cipher_text
def download_m3u8_video(url,  headers:dict,downloadFolder:str, fileName:str , max_workers=10):
    time = datetime.datetime.now() 
    tempFolder=os.path.join(downloadFolder,"temp",time.strftime('%H-%M-%S.%f'))
    try:
        if not os.path.exists(tempFolder):
            os.makedirs(tempFolder)  
        real_url = get_real_url(url,headers)
        # 未加密
        key=None
        if type (real_url) == m3u8.M3U8:
            playlist= real_url 
        else:
            playlist = m3u8.load(uri=real_url, headers=headers)
            key = requests.get(playlist.keys[-1].uri, headers=headers).content

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, seg in enumerate(playlist.segments):
                log.warn(f"下载{i}->{seg.absolute_uri}...")
                pool.submit(download_ts, seg.absolute_uri, key, i, headers, tempFolder)

        with open(os.path.join(downloadFolder,fileName), 'wb') as fw:
            files = glob.glob(tempFolder+'/*.ts')
            for file in files:
                with open(file, 'rb') as fr:
                    fw.write(fr.read())
                    print(f'\r{file}已合并!总数:{len(files)}', end=" ")
                os.remove(file) #删除临时文件 
    finally:
        files = glob.glob(tempFolder+'/*.ts')
        for file in files: 
            os.remove(file) #删除临时文件 
        os.rmdir(tempFolder)  #删除临时文件夹
def createHeader( referer:str):
    header={
        "Referer": referer, 
        "User-Agent": 'Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.109 Safari/537.36 CrKey/1.54.248666'
    }
    return header;
def download(url:str, header:dict, downloadFolder:str, fileName:str,timeout:int=60)->None: 
    try:
        log.info("下载开始...")
        status,data=http.download(requests.utils.requote_uri(url),timeout=timeout,header_dict=header,verify=False) 
        if status == 200 or status == 206:
            log.warn(os.path.join(downloadFolder,fileName))
            File=open(os.path.join(downloadFolder,fileName),"wb") # Opens the file for writing.
            File.write(data)
            log.end_mark(f"下载{url}完成.")
    except  Exception as e:
        log.err(e) 
    finally:
        log.info("结束下载.")
    
if __name__ == '__main__' :
    default_save_dir=os.path.join(os.path.abspath("."),"Download") 
    parser=argparse.ArgumentParser(description="下载文件")
    parser.add_argument("-u","--url",type=str, help="需要下载的URL地址")
    parser.add_argument('--m3u8' ,default=False, action=argparse.BooleanOptionalAction,help=f"is m3u9")
    parser.add_argument("-r","--referer",type=str, help="referer")

    parser.add_argument("-p","--proxy",type=str, help="代理服务器,ip:port",default=None)  
    parser.add_argument("-d","--folder",type=str,help=f"保存目录 [{default_save_dir}]",default=default_save_dir)
    parser.add_argument("-f","--file_name",type=str,help=f"文件名" )
    parser.add_argument("-t","--timeout",type=int,help="http 超时时间",default=15)
    args=parser.parse_args() 
    if args.proxy !=None:
        arr = args.proxy.split(":")
        set_proxy(arr[0], int(arr[1]))
    if args.url == None:
        parser.print_help()
        sys.exit(0)
    header=createHeader(args.referer)
    log.warn(f"is m3u8:{args.m3u8}")
    if args.m3u8:
        download_m3u8_video(args.url,header,args.folder,args.file_name)
    else:
        download(args.url, header, args.folder, args.file_name,args.timeout)
    