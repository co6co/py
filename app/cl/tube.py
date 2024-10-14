import time
from pytubefix import YouTube, Stream, Caption
from pytubefix.cli import on_progress, display_progress_bar
from pytubefix import Playlist
from co6co import getByteUnit
from typing import List
from co6co.task.pools import limitThreadPoolExecutor
from concurrent.futures import Future
from co6co.utils.log import progress_bar
import signal
# url = "https://www.youtube.com/watch?v=lxOFGvHBsTY"
# proxys = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
proxys = {"http": "http://127.0.0.1:9667", "https": "http://127.0.0.1:9667"}

 
def getInt(tip):
    try:
        c = input(tip)
        return int(c)
    except:
        return -1


class DownLoad:
    streams: List[Stream] = None
    captions: List[Caption] = None
    title: str = None

    @staticmethod
    def on_progress(stream: Stream,
                    chunk: bytes,
                    bytes_remaining: int
                    ) -> None:  # pylint: disable=W0613
        if not hasattr(stream, "downByts"):
            stream.downByts = len(chunk)
        else:
            stream.downByts += len(chunk)
        sum = stream.downByts+bytes_remaining
        print(f'download {getByteUnit(len(chunk))}, downloading {round(stream.downByts/sum, 4)*100}% {getByteUnit(stream.downByts)}/{getByteUnit(sum)}  remainng:{getByteUnit(bytes_remaining)} data to file .. ')
        filesize = stream.filesize
        bytes_received = filesize - bytes_remaining
        display_progress_bar(bytes_received, filesize)

    @staticmethod
    def on_pro(stream: Stream, bytes, bytes_remaining):
        if not hasattr(stream, "downByts"):
            stream.downByts = len(bytes)
        else:
            stream.downByts += len(bytes)
        progress_bar(stream.downByts/stream.filesize, "stream,剩余：{}-{}".format(getByteUnit(bytes_remaining), stream.filesize))

    def __init__(self, url: str = None, title=None, streams: List[Stream] = None, captions: List[Caption] = None) -> None:
        if url == None:
            self.title = title
            self.streams = streams
            self.captions = captions
        else:
            yt = YouTube(url, on_progress_callback=DownLoad. on_pro, proxies=proxys)
            # yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, on_progress_callback=on_progress, proxies=proxys)
            print("视频标题", yt.title)
            self.title = yt.title
            self.streams = yt.streams.all()
            self.captions = yt.captions.all()
        pass

    def print(self):
        for item in self.streams:
            print(item)

    def downloadVideo(self, itag: int):
        """
        下载视频或音频
        """
        checkedList = [i for i in self.streams if i.itag == itag]
        for stream in checkedList:
            stream.download()

    def downCaption(self):
        caption = self.captions
        while True:
            caplist = [str(cap)for cap in caption]
            print("\n".join(caplist))
            # caption = yt.captions.get_by_language_code('en')
            code = input("caption code,q:quit:")
            if code == 'q':
                break

            checkedList = [i for i in caption if i.code == code]
            for checked in checkedList:
                file = input("保存路径:")
                checked.download(title=self.title, output_path=file)

            # caption.save_captions("captions.txt")


class downPlaylist:
    title: str
    playList: Playlist
    urlList: List[str]
    quitFlag: bool
    downingArr: List[Future]

    def _result(self, f: Future):
        exception = f.exception()
        if not exception:
            # 如果获取不到异常说明破解成功
            if not self.quitFlag:
                print("{}=>'{}',下载完成".format(f.url,   f.title))
        else:
            # 如果获取不到异常说明破解成功
            print("{}=>'{}',下载完成错误->{}".format(f.url, f.title, exception))
        self.downingArr.remove(f)

    def __init__(self, url) -> None:
        playList = Playlist(url, proxies=proxys)
        print("列表标题", playList.title, len(playList))
        self.title = playList.title
        self.urlList = [i for i in playList]
        self.playList = playList
        self.quitFlag = False
        self.downingArr = []
        pass

    def start(self, worker: int = None):
        c = None
        # Register the custom handler for SIGINT
        signal.signal(signal.SIGINT, self.custom_handler)
        pool = limitThreadPoolExecutor(max_workers=worker, thread_name_prefix="download")

        for index, video in enumerate(self. playList.videos):
            if self.quitFlag:
                print("用户主动退出下载，等待下载的任务..")
                count = len(self.downingArr)
                while len(self.downingArr) > 0:
                    time.sleep(1)
                    progress_bar((count-len(self.downingArr))/count, "等待已创建的任务完成")

                break
            print(index, "=>", self.urlList[index])
            down = DownLoad(title=video.title, streams=video.streams, captions=video.captions)
            if c == None:
                down.print()
                c = getInt("下载itag,只有第一次会出现后面按第一次选择的下载:")

            f = pool.submit(down.downloadVideo, c)
            f.pool = pool
            f.title = video.title
            f.url = self.urlList[index]
            f.index = index
            self.downingArr.append(f)
            f.add_done_callback(self._result)

    def custom_handler(self, signum, frame):
        print("Caught signal:", signum)
        self.quitFlag = True


if __name__ == "__main__":
    url = input("输入要下载的URL:")
    while True:
        c = getInt("下载类型:0->列表,1-> 音视频,2->字幕,9->退出:")
        if c == 0:
            download = downPlaylist(url)
            download.start()
        elif c == 1:
            down = DownLoad(url)
            while True:
                down.print()
                c = getInt("请输入itag:")
                if c == -1:
                    break
                down.downloadVideo(c)
        elif c == 2:
            down = DownLoad(url)
            down.downCaption()
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
