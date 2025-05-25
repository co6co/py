import time
from pytubefix import YouTube, Stream, Caption,StreamQuery
from pytubefix.cli import on_progress, display_progress_bar
from pytubefix import Playlist
from co6co import getByteUnit
from co6co.utils import convert_size
from typing import List
from co6co.task.pools import limitThreadPoolExecutor
from concurrent.futures import Future
from co6co.utils.log import progress_bar,err,warn
import signal
from typing import Callable
import os
# url = "https://www.youtube.com/watch?v=lxOFGvHBsTY"
proxys = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
#proxys = {"http": "http://127.0.0.1:9667", "https": "http://127.0.0.1:9667"}
#proxys = {"http": "http://127.0.0.1:9666", "https": "http://127.0.0.1:9666"}
 
def getInt(tip,default=-1):
    try:
        c = input(tip)
        return int(c)
    except:
        return default
 
def download_with_resume(stream :Stream, output_path=None, filename=None):
    file_size = stream.filesize 
    if output_path is None:
        output_path = os.getcwd()
    if filename is None:
        filename = stream.default_filename
    #stream.title
    #stream.mime_type
    print("下载文件名：",filename)
    file_path = os.path.join(output_path, filename)
    temp_file = file_path + '.part'
    
    # 检查是否有临时文件（即是否有未完成的下载）
    downloaded = 0
    if os.path.exists(temp_file):
        downloaded = os.path.getsize(temp_file)
        if downloaded < file_size:
            print(f"从 {downloaded} 字节开始继续下载...")
        else:
            # 如果临时文件大小等于或大于文件总大小，删除临时文件重新下载
            os.remove(temp_file)
            downloaded = 0
    
    # 如果已经下载完成，直接返回
    if downloaded == file_size:
        if os.path.exists(temp_file):
            os.rename(temp_file, file_path)
        print(f"文件已下载完成: {file_path}")
        return file_path
    
    # 开始下载或继续下载
    headers = {'Range': f'bytes={downloaded}-'} if downloaded else {}
    
    with requests.get(stream.url, headers=headers, stream=True) as r:
        r.raise_for_status()
        mode = 'ab' if downloaded else 'wb'
        with open(temp_file, mode) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # 下载完成后，将临时文件重命名为最终文件名
    os.rename(temp_file, file_path)
    print(f"下载完成: {file_path}")
    return file_path

class ResumableStream(Stream):
    def create(stream:Stream,yt:YouTube):
        stream = ResumableStream( stream_dict=stream.__dict__, monostate=stream._monostate)
        return stream
    def __init__(self, stream, monostate, chunk_size=8192):
        super().__init__(stream, monostate)
        self.chunk_size = chunk_size
    def set_chunk_size(self, size):
        """设置下载块大小
        Args:
            size: 新的块大小（字节）
        """
        self.chunk_size = size

    def download(self, output_path=None, filename=None, skip_existing=True):
        file_size = self.filesize
        
        if output_path is None:
            output_path = os.getcwd()
        if filename is None:
            filename = self.default_filename
            
        file_path = os.path.join(output_path, filename)
        temp_file = file_path + '.part'
        
        # 检查是否有临时文件（即是否有未完成的下载）
        downloaded = 0
        if os.path.exists(temp_file):
            downloaded = os.path.getsize(temp_file)
            if downloaded < file_size:
                print(f"从 {downloaded} 字节开始继续下载...")
            else:
                # 如果临时文件大小等于或大于文件总大小，删除临时文件重新下载
                os.remove(temp_file)
                downloaded = 0
        
        # 如果已经下载完成，直接返回
        if downloaded == file_size:
            if os.path.exists(temp_file):
                os.rename(temp_file, file_path)
            print(f"文件已下载完成: {file_path}")
            return file_path
        
         # 开始下载或继续下载
        headers = {'Range': f'bytes={downloaded}-'} if downloaded else {} 
        try:
            with requests.get(self.url, headers=headers,proxies=proxys, stream=True, timeout=10) as r:
                r.raise_for_status()
                mode = 'ab' if downloaded else 'wb'
                with open(temp_file, mode) as f:
                    # 使用自定义块大小进行下载
                    for chunk in r.iter_content(chunk_size=self.chunk_size):
                        if chunk:  # 过滤掉空块
                            f.write(chunk)
        except Exception as e:
            print(f"下载过程中发生错误: {e}")
            return None
        
        # 下载完成后，将临时文件重命名为最终文件名
        os.rename(temp_file, file_path)
        print(f"下载完成: {file_path} (块大小: {self.chunk_size} 字节)")
        return file_path
class DownLoad:
    streams: List[Stream] = None
    captions: List[Caption] = None
    title: str = None
 
    def on_progress(self,stream: Stream,
                    chunk: bytes,
                    bytes_remaining: int
                    ) -> None:  # pylint: disable=W0613
        if not hasattr(stream, "downByts"):
            stream.downByts = len(chunk)
        else:
            stream.downByts += len(chunk)
        sum = stream.downByts+bytes_remaining 
        print(f'download {convert_size(len(chunk))}, downloading {round(stream.downByts/sum*100, 2)}% {convert_size(stream.downByts)}/{convert_size(sum)}  remainng:{convert_size(bytes_remaining)} data to file .. ')
        filesize = stream.filesize
        bytes_received = filesize - bytes_remaining
        display_progress_bar(bytes_received, filesize)
 
    def on_pro(self,stream: Stream, bytes, bytes_remaining):
        if not hasattr(stream, "downByts"):
            stream.downByts = len(bytes)
        else:
            stream.downByts += len(bytes)
        progress_bar(stream.downByts/stream.filesize, "stream:{}\t完成/剩余/总：{}/{}/{}".format(self.title, convert_size(stream.downByts),convert_size(bytes_remaining), convert_size(stream.filesize)))

    @staticmethod
    def createYoube(url:str): 
        # 方法1: 使用PO令牌机制
        yt = YouTube(url, proxies=proxys )
        
        # 方法2: 使用WEB客户端模式 (如果库支持)
        #yt = YouTube(url, on_progress_callback=DownLoad.on_pro, proxies=proxys, client="web")
        return yt
    def __init__(self, yt:YouTube  ) -> None:
        yt.register_on_progress_callback(self.on_pro)
        self.downloadFolder ="D:\\temp\\"
        if not os.path.exists(self.downloadFolder):
            os.makedirs(self.downloadFolder) 
        print("视频标题", yt.title)
        self.title = yt.title
        self.youtTube =yt
        #self.streams = yt.streams.all()
        #self.captions = yt.captions.all() 
        #self.streams:StreamQuery = yt.streams
        #self.captions =yt. captions
           
        pass

    def print(self):
        #https://www.youtube.com/playlist?list=PLgjl5F_IQpFfv48q3aRChUfETXGafDR9z
        for item in self.youtTube.streams.all():
            print(item)
    def download(self,itag: int):  
        if itag==0:
            self._downLoadHighestResolution()
        else:
            self._downloadVideo(itag)
    def _downloadVideo(self, itag: int):
        """
        下载视频或音频
        """
        
        checkedList  = [i for i in self.youtTube.streams.all() if i.itag == itag]
        for stream in checkedList: 
            #self.youtTube.fmt_streams
            #stream_dict= self.youtTube.streams.get_highest_resolution().stream_dict
            #ResumableStream(stream.to_dict)
            filePath=stream.download(output_path=self.downloadFolder,filename_prefix=itag, max_retries=100,timeout=600,skip_existing=False)
            #filePath=download_with_resume(stream,self.downloadFolder)
            print(f"{self.title},保存到{filePath}")
    def _downLoadHighestResolution(self ):
        """
        高清流
        """
        try: 
            stream = self.youtTube.streams.get_highest_resolution()
            stream.download(output_path=self.downloadFolder,skip_existing=False)
        except Exception as e:
            print("下载异常",e)
            raise e
    def printCaption(self):
        """
        打印字母编码
        """
        caption = self.youtTube.captions
        caplist = [str(cap)for cap in caption]
        print("\n".join(caplist))
    def downCaptionOne(self,code:str):
        """
        下载指定字幕
        """
        caption = self.youtTube.captions
        checkedList = [i for i in caption if i.code == code]
        for checked in checkedList: 
             checked.download(title=self.title, output_path=self.downloadFolder )

    def downCaption(self):
        caption = self.youtTube.captions
        while True:
            self.printCaption() 
            # caption = yt.captions.get_by_language_code('en')
            code = input("caption code,q:quit:")
            if code == 'q':
                break 
            checkedList = [i for i in caption if i.code == code]
            for checked in checkedList: 
                checked.download(title=self.title, filename_prefix=self.downloadFolder)

            # caption.save_captions("captions.txt")


class downPlaylist:
    title: str
    playList: Playlist
    urlList: List[str]
    quitFlag: bool
    downingArr: List[Future]
    errorList:list[int]
    downCategory:int
    """
    0: 音视频
    1: 字幕
    """
    streamItag:int
    """
    0: 高清流
    >0: 流Itag
    """
    captionCode:str
    """
    字幕代码
    """
    def error( self,index:int,url:str,e:Exception):
        if index not in self.errorList :
            self.errorList.append(index)
        err("{}=>{}下载完成异常{}".format(index,url,e), self.errorList)

    def _result(self, f: Future ):
        exception = f.exception()
        if not exception:
            # 如果获取不到异常说明破解成功
            if not self.quitFlag:
                print("\n{}=>{}=>'{}',下载完成".format(f.index,f.url,   f.title)) 
            if f.errFlag:
                print("移除ErrorList",f.index,"...")
                self.errorList.remove(f.index) 
                print("移除ErrorList,剩余：",self.errorList)
        else:
            # 如果获取不到异常说明破解成功 
            self.error(f.index,f.url,exception) 
        self.downingArr.remove(f)

    def __init__(self, url) -> None:
        playList = Playlist(url, proxies=proxys)
        print("列表标题", playList.title, len(playList))
        self.title = playList.title
        self.urlList = [i for i in playList]
        self.playList = playList
        self.quitFlag = False
        self.downingArr = []
        self.errorList=[] 
        self.downCategory=-1
        self.streamItag=-1
        self.captionCode=None

        pass
    def downOne(self,index,yt:YouTube,pool:limitThreadPoolExecutor, errorDown:bool=None):
        """
        返回Itag,
        raturn Itag or captionCode
        """  
        try: 
            print(index, "=>", self.urlList[index],isinstance(yt,YouTube))
            down = DownLoad(yt) 
            # 下载类型 
            if self.downCategory==-1:
                self.downCategory = getInt("下载类型:0->音视频,1->字幕:") 
            # 字幕
            if not self.captionCode and self.downCategory==1:
                down.printCaption()
                self.captionCode = input("字幕编码：")
            #音视频
            if  self.downCategory==0 and self.streamItag==-1: 
                down.print()
                self.streamItag = getInt("下载itag流, 0:下载高清流:") 
          
            pararms=(down.download,self.streamItag) if self.downCategory==0 else (down.downCaptionOne,self.captionCode)
         
            f = pool.submit(*pararms)

            f.pool = pool
            f.title = yt.title
            f.url = self.urlList[index]
            f.index = index
            f.errFlag=errorDown
            self.downingArr.append(f)
            f.add_done_callback(self._result)  
        except Exception as e: 
            self.error(index,self.urlList[index],e) 


    def start(self, worker: int = None,itag:int=None,errorDown=False): 
        # Register the custom handler for SIGINT
        signal.signal(signal.SIGINT, self.custom_handler)
        pool = limitThreadPoolExecutor(max_workers=worker, thread_name_prefix="download") 
        if not errorDown:
            max=len(self.playList.videos)-1
            s = getInt(f"indexStartValue:0-{max}:<==",0)
            e = getInt(f"indexEndValue:{max}:<==",max)
       
        for index, youTube in enumerate(self.playList.videos): 
            youTube.use_po_token=True
            if errorDown and index not in self.errorList:continue  
            if not errorDown and (index<s or index >e) :continue
            if self.quitFlag:
                print("用户主动退出下载，等待下载的任务..")
                count = len(self.downingArr)
                while len(self.downingArr) > 0:
                    time.sleep(1)
                    progress_bar((count-len(self.downingArr))/count, "等待已创建的任务完成") 
                break
            self.downOne(index,youTube,pool,errorDown)
        while len(self.downingArr) and not self.quitFlag:
            time.sleep(10)
            print("downloading ...，剩余下载数->",len(self.downingArr),self.downingArr,"errorList->",len(self.errorList),self.errorList)
        if not self.quitFlag and  len(self.errorList)>0:
            warn("开始下载有异常的数据:len-->{}".format(len(self.errorList)))
            self.start(itag=itag,errorDown=True)
            
    def custom_handler(self, signum, frame):
        print("Caught signal:", signum)
        self.quitFlag = True

import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
def create_proxy_session(proxy_address, retry=3):
    session = requests.Session() 
    # 设置SOCKS5代理 (示例: socks5://user:pass@host:port)
    session.proxies = {
        "http": proxy_address,
        "https": proxy_address
    }
    
    # 配置重试策略
    retries = Retry(
        total=retry,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries)) 
    return session

def demo(url):  
    """
    socks --> Missing dependencies for SOCKS support
    # install PySocks
    pip install PySocks
    """
    import requests
    session=create_proxy_session("socks5://127.0.0.1:10808")
    r = session.get("https://api.ipify.org?format=json" )
    print("应显示代理IP", r.json())  # 应显示代理IP

    yt = YouTube(url, proxies=proxys )
    print(yt.title) 
    listStream=yt.streams.filter(progressive=True, mime_type=None).order_by("resolution")
    print(*listStream)
    allStream=yt.streams.order_by("resolution").all()
    print("allStream->")
    print( *allStream,sep='\n')
    #stream = yt.streams.get_highest_resolution()
    #stream.download() 
if __name__ == "__main__":
    url = input("输入要下载的URL:")
    while True:
        c = getInt("下载类型:0->列表,1-> 音视频,2->字幕,8->重新输入url,9->退出:")
        if c == 0:
            download = downPlaylist(url)
            download.start()
        elif c == 1: 
            down = DownLoad(DownLoad.createYoube(url))
            while True: 
                down.print()
                c = getInt("请输入itag,[0->高清流]:")
                if c == -1:
                    break
                down.download(c)
        elif c == 2:
            down = DownLoad(url)
            down.downCaption()
        elif c==3:
            demo(url)
            pass 

        elif c==8:
            url= input("输入要下载的URL:")
        elif c == 9:
            break


'''
ys = yt.streams.get_highest_resolution()
ys.download()
# 下载音频
ys = yt.streams.get_audio_only()
ys.download(mp3=True)

pl = Playlist(url)
for video in pl.videos:
    ys = video.streams.get_audio_only()
    ys.download(mp3=True)

# 通道
from pytubefix import Channel
c = Channel("https://www.youtube.com/@ProgrammingKnowledge")
print(f'Downloading videos by: {c.channel_name}')
for video in c.videos:
    download = video.streams.get_highest_resolution().download()
# 播放列表
from pytubefix import Playlist
from pytubefix.cli import on_progress
 
url = "url"

pl = Playlist(url)

for video in pl.videos:
    ys = video.streams.get_audio_only()
    ys.download(mp3=True)
'''
