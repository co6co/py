
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from urllib.error import HTTPError
import time
from pytubefix import YouTube, Stream, Caption, StreamQuery
from pytubefix.cli import on_progress, display_progress_bar
from pytubefix import Playlist, request
from pytubefix.streams import logger
from co6co import getByteUnit
from co6co.utils import convert_size, log
from typing import List
from co6co.task.pools import limitThreadPoolExecutor
from concurrent.futures import Future
from co6co.utils.log import progress_bar, err, warn
import signal
from typing import Callable, Optional
import os
import socket
# url = "https://www.youtube.com/watch?v=lxOFGvHBsTY"
proxys = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
# proxys = {"http": "http://127.0.0.1:9667", "https": "http://127.0.0.1:9667"}
# proxys = {"http": "http://127.0.0.1:9666", "https": "http://127.0.0.1:9666"}


def stream(url,
           timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
           max_retries=0, downloaded: int = 0, file_size: int = 0):
    """Read the response in chunks.
    :param str url: The URL to perform the GET request for.
    :rtype: Iterable[bytes]
    """
    if not file_size:
        file_size: int = request.default_range_size  # fake filesize to start
    log.warn(f"文件大小：{downloaded}/{file_size}")
    while downloaded < file_size:
        stop_pos = min(downloaded + request. default_range_size, file_size) - 1
        range_header = f"bytes={downloaded}-{stop_pos}"
        tries = 0

        # Attempt to make the request multiple times as necessary.
        while True:
            # If the max retries is exceeded, raise an exception
            if tries >= 1 + max_retries:
                raise request. MaxRetriesExceeded()

            # Try to execute the request, ignoring socket timeouts
            try:
                response = request. _execute_request(
                    f"{url}&range={downloaded}-{stop_pos}",
                    method="GET",
                    timeout=timeout
                )
            except request.URLError as e:
                # We only want to skip over timeout errors, and
                # raise any other URLError exceptions
                if not isinstance(e.reason, socket.timeout):
                    raise
            except request.http.client.IncompleteRead:
                # Allow retries on IncompleteRead errors for unreliable connections
                pass
            else:
                # On a successful request, break from loop
                break
            tries += 1

        if file_size == request.default_range_size:
            try:
                resp = request._execute_request(
                    f"{url}&range=0-99999999999",
                    method="GET",
                    timeout=timeout
                )
                content_range = resp.info()["Content-Length"]
                file_size = int(content_range)
            except (KeyError, IndexError, ValueError) as e:
                logger.error(e)
        while True:
            try:
                chunk = response.read()
            except StopIteration:
                return

            if not chunk:
                break

            if chunk:
                downloaded += len(chunk)
            yield chunk
    return  # pylint: disable=R1711


request.stream = stream


def StreamDownload(self: Stream,
                   output_path: Optional[str] = None,
                   filename: Optional[str] = None,
                   filename_prefix: Optional[str] = None,
                   skip_existing: bool = True,
                   timeout: Optional[int] = None,
                   max_retries: Optional[int] = 0,
                   mp3: bool = False,
                   remove_problematic_character: str = None) -> str:
    print("00000")
    if remove_problematic_character:
        filename = self.title.replace(remove_problematic_character, "")

    if mp3:
        if filename is None:
            filename = self.title + ".mp3"
        else:
            filename = filename + ".mp3"

    file_path = self.get_file_path(
        filename=filename,
        output_path=output_path,
        filename_prefix=filename_prefix,
    )
    # 获取本地文件已存在的大小
    # if self.exists_at_path(file_path):
    #    logger.debug(f'file {file_path} already exists, skipping')
    #    self.on_complete(file_path)
    #    return file_path
    existing_size = 0
    if os.path.exists(file_path):
        existing_size = os.path.getsize(file_path)
    print(f"文件大小为 {existing_size}/{self.filesize} 字节")
    # 计算剩余需要下载的字节数
    bytes_remaining = self.filesize - existing_size if self.filesize else None
    with open(file_path, "ab") as fh:
        try:
            if existing_size:
                data = bytes()
                # 不写入字节
                self.on_progress(data, fh, bytes_remaining)
            for chunk in request.stream(
                self.url,
                timeout=timeout,
                max_retries=max_retries,
                downloaded=existing_size,
                file_size=self.filesize
            ):
                # reduce the (bytes) remainder by the length of the chunk.
                bytes_remaining -= len(chunk)
                # send to the on_progress callback.
                self.on_progress(chunk, fh, bytes_remaining)
        except HTTPError as e:
            if e.code != 404:
                raise
        except StopIteration:
            # Some adaptive streams need to be requested with sequence numbers
            for chunk in request.seq_stream(
                self.url,
                timeout=timeout,
                max_retries=max_retries
            ):
                # reduce the (bytes) remainder by the length of the chunk.
                bytes_remaining -= len(chunk)
                # send to the on_progress callback.
                self.on_progress(chunk, fh, bytes_remaining)

    self.on_complete(file_path)
    return file_path


"""
# ++++++
        # 获取本地文件已存在的大小
        existing_size = 0
        if os.path.exists(file_path):
            existing_size = os.path.getsize(file_path)
            logger.debug(f"本地文件已存在，大小为 {existing_size} 字节")
        # 计算剩余需要下载的字节数
        bytes_remaining =self.filesize - existing_size if self.filesize else None
        #print("文件大小",bytes_remaining)
        
        #++++++
        with open(file_path, "ab") as fh:
            try: 
                #if existing_size:
                #    self.on_progress(existing_size, fh, bytes_remaining)

                
# request
def stream(url,
           timeout=socket._GLOBAL_DEFAULT_TIMEOUT, 
           max_retries=0,downloaded:int=0):
     
    file_size: int = default_range_size   # fake filesize to start 
    first=True
    while first or downloaded < file_size: 
        stop_pos = min(downloaded + default_range_size, file_size) - 1
        if  first:
            first=False
            stop_pos=(downloaded + default_range_size)-1
        else:
            stop_pos = min(downloaded + default_range_size, file_size) - 1
        range_header = f"bytes={downloaded}-{stop_pos}"
"""


def getInt(tip, default=-1):
    try:
        c = input(tip)
        return int(c)
    except:
        return default


class DownLoad:
    streams: List[Stream] = None
    captions: List[Caption] = None
    title: str = None

    def on_progress(self, stream: Stream,
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

    def on_pro(self, stream: Stream, bytes, bytes_remaining):
        if not hasattr(stream, "downByts"):
            stream.downByts = len(bytes)
        else:
            stream.downByts += len(bytes)
        progress_bar(stream.downByts/stream.filesize, "stream:{}\t完成/剩余/总：{}/{}/{}".format(self.title, convert_size(stream.downByts), convert_size(bytes_remaining), convert_size(stream.filesize)))

    @staticmethod
    def createYoube(url: str):
        # 方法1: 使用PO令牌机制
        #/YunzheZJU/youtube-po-token-generator
        yt = YouTube(url, proxies=proxys, use_po_token=True) #, use_po_token=True

        # 方法2: 使用WEB客户端模式 (如果库支持)
        # yt = YouTube(url, on_progress_callback=DownLoad.on_pro, proxies=proxys, client="web")
        return yt

    def __init__(self, yt: YouTube) -> None:
        yt.register_on_progress_callback(self.on_pro)
        self.downloadFolder = "D:\\temp\\"
        if not os.path.exists(self.downloadFolder):
            os.makedirs(self.downloadFolder)
        print("视频标题", yt.title)
        self.title = yt.title
        self.youtTube = yt
        # self.streams = yt.streams.all()
        # self.captions = yt.captions.all()
        # self.streams:StreamQuery = yt.streams
        # self.captions =yt. captions

        pass

    def print(self):
        # https://www.youtube.com/playlist?list=PLgjl5F_IQpFfv48q3aRChUfETXGafDR9z
        for item in self.youtTube.streams.all():
            print(item)

    def download(self, itag: int):
        if itag == 0:
            self._downLoadHighestResolution()
        else:
            self._downloadVideo(itag)

    def _downloadVideo(self, itag: int):
        """
        下载视频或音频
        """
        checkedList = [i for i in self.youtTube.streams.all() if i.itag == itag]
        for stream in checkedList:
            try:
                # self.youtTube.fmt_streams
                # stream_dict= self.youtTube.streams.get_highest_resolution().stream_dict
                stream.download = StreamDownload.__get__(stream, Stream)  # 绑定方法到实例
                filePath = stream.download(output_path=self.downloadFolder, filename_prefix=itag, max_retries=100, timeout=600, skip_existing=False)

                print(f"{self.title},保存到{filePath}")
            except Exception as e:
                print("下载异常", e)
                log.err("ddd", e)

    def _downLoadHighestResolution(self):
        """
        高清流
        """
        try:
            stream = self.youtTube.streams.get_highest_resolution()
            stream.download = StreamDownload.__get__(stream, Stream)  # 绑定方法到实例
            stream.download(output_path=self.downloadFolder, skip_existing=False)
        except Exception as e:
            print("下载异常", e)
            log.err("ddd", e)
            raise e

    def printCaption(self):
        """
        打印字母编码
        """
        caption = self.youtTube.captions
        caplist = [str(cap)for cap in caption]
        print("\n".join(caplist))

    def downCaptionOne(self, code: str):
        """
        下载指定字幕
        """
        caption = self.youtTube.captions
        checkedList = [i for i in caption if i.code == code]
        for checked in checkedList:
            checked.download(title=self.title, output_path=self.downloadFolder)

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
    errorList: list[int]
    downCategory: int
    """
    0: 音视频
    1: 字幕
    """
    streamItag: int
    """
    0: 高清流
    >0: 流Itag
    """
    captionCode: str
    """
    字幕代码
    """

    def error(self, index: int, url: str, e: Exception):
        if index not in self.errorList:
            self.errorList.append(index)
        err("{}=>{}下载完成异常{}".format(index, url, e), self.errorList)

    def _result(self, f: Future):
        exception = f.exception()
        if not exception:
            # 如果获取不到异常说明破解成功
            if not self.quitFlag:
                print("\n{}=>{}=>'{}',下载完成".format(f.index, f.url,   f.title))
            if f.errFlag:
                print("移除ErrorList", f.index, "...")
                self.errorList.remove(f.index)
                print("移除ErrorList,剩余：", self.errorList)
        else:
            # 如果获取不到异常说明破解成功
            self.error(f.index, f.url, exception)
        self.downingArr.remove(f)

    def __init__(self, url) -> None:
        playList = Playlist(url, proxies=proxys)
        print("列表标题", playList.title, len(playList))
        self.title = playList.title
        self.urlList = [i for i in playList]
        self.playList = playList
        self.quitFlag = False
        self.downingArr = []
        self.errorList = []
        self.downCategory = -1
        self.streamItag = -1
        self.captionCode = None

        pass

    def downOne(self, index, yt: YouTube, pool: limitThreadPoolExecutor, errorDown: bool = None):
        """
        返回Itag,
        raturn Itag or captionCode
        """
        try:
            print(index, "=>", self.urlList[index], isinstance(yt, YouTube))
            down = DownLoad(yt)
            # 下载类型
            if self.downCategory == -1:
                self.downCategory = getInt("下载类型:0->音视频,1->字幕:")
            # 字幕
            if not self.captionCode and self.downCategory == 1:
                down.printCaption()
                self.captionCode = input("字幕编码：")
            # 音视频
            if self.downCategory == 0 and self.streamItag == -1:
                down.print()
                self.streamItag = getInt("下载itag流, 0:下载高清流:")

            pararms = (down.download, self.streamItag) if self.downCategory == 0 else (down.downCaptionOne, self.captionCode)

            f = pool.submit(*pararms)

            f.pool = pool
            f.title = yt.title
            f.url = self.urlList[index]
            f.index = index
            f.errFlag = errorDown
            self.downingArr.append(f)
            f.add_done_callback(self._result)
        except Exception as e:
            self.error(index, self.urlList[index], e)

    def start(self, worker: int = None, itag: int = None, errorDown=False):
        # Register the custom handler for SIGINT
        signal.signal(signal.SIGINT, self.custom_handler)
        pool = limitThreadPoolExecutor(max_workers=worker, thread_name_prefix="download")
        if not errorDown:
            max = len(self.playList.videos)-1
            s = getInt(f"indexStartValue:0-{max}:<==", 0)
            e = getInt(f"indexEndValue:{max}:<==", max)

        for index, youTube in enumerate(self.playList.videos):
            youTube.use_po_token = True
            if errorDown and index not in self.errorList:
                continue
            if not errorDown and (index < s or index > e):
                continue
            if self.quitFlag:
                print("用户主动退出下载，等待下载的任务..")
                count = len(self.downingArr)
                while len(self.downingArr) > 0:
                    time.sleep(1)
                    progress_bar((count-len(self.downingArr))/count, "等待已创建的任务完成")
                break
            self.downOne(index, youTube, pool, errorDown)
        while len(self.downingArr) and not self.quitFlag:
            time.sleep(10)
            print("downloading ...，剩余下载数->", len(self.downingArr), self.downingArr, "errorList->", len(self.errorList), self.errorList)
        if not self.quitFlag and len(self.errorList) > 0:
            warn("开始下载有异常的数据:len-->{}".format(len(self.errorList)))
            self.start(itag=itag, errorDown=True)

    def custom_handler(self, signum, frame):
        print("Caught signal:", signum)
        self.quitFlag = True


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
    session = create_proxy_session("socks5://127.0.0.1:10808")
    r = session.get("https://api.ipify.org?format=json")
    print("应显示代理IP", r.json())  # 应显示代理IP

    yt = YouTube(url, proxies=proxys)
    print(yt.title)
    listStream = yt.streams.filter(progressive=True, mime_type=None).order_by("resolution")
    print(*listStream)
    allStream = yt.streams.order_by("resolution").all()
    print("allStream->")
    print(*allStream, sep='\n')

    # stream = yt.streams.get_highest_resolution()
    # stream.download()
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
        elif c == 3:
            demo(url)
            pass

        elif c == 8:
            url = input("输入要下载的URL:")
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
