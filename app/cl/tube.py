from pytubefix import YouTube,Stream
from pytubefix.cli import on_progress,display_progress_bar
from pytubefix import Playlist
from co6co import getByteUnit 
url = input("输入要下载的URL:")
#url = "https://www.youtube.com/watch?v=lxOFGvHBsTY"
proxys = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}


def on_progress(stream: Stream,
                chunk: bytes,
                bytes_remaining: int
                ) -> None:  # pylint: disable=W0613
    if not hasattr(stream,"downByts")  : stream.downByts=len(chunk)
    else :stream.downByts+=len(chunk)  
    sum=stream.downByts+bytes_remaining
    print( f'download {getByteUnit(len( chunk))}, downloading {round(stream.downByts/sum,4)*100}% {getByteUnit(stream.downByts)}/{getByteUnit(sum)}  remainng:{getByteUnit(bytes_remaining )} data to file .. ')
    filesize = stream.filesize
    bytes_received = filesize - bytes_remaining
    display_progress_bar(bytes_received, filesize)
yt = YouTube(url, on_progress_callback=on_progress, proxies=proxys)
#yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, on_progress_callback=on_progress, proxies=proxys)
print("标题：", yt.title)
streamList = yt.streams.all()


def getInt(tip):
    try:
        c=input(tip)
        return int(c)
    except:
        return -1

def downloadVideo():
    """
    下载视频或音频
    """ 
    def checkItem():
        """
        选择要下载的序号
        """
        for item in streamList:
            print(item)
        itag = getInt("input itag,-1->quit:")
        if itag == -1:
            return None
        return itag
    while True:
        itag = checkItem()
        if itag == None:
            break
        checkedList = [i for i in streamList if i.itag == itag]

        for checked in checkedList:
            checked.download() 


def downCaption():
    caption = yt.captions.all()
    while True:
        caplist=[str(cap)for cap in caption]
        print("\n".join( caplist) )
        # caption = yt.captions.get_by_language_code('en')
        code = input("caption code,q:quit:")
        if code=='q':
            break

        checkedList = [i for i in caption if i.code == code]
        for checked in checkedList:
            file = input("保存路径:")
            checked.download(title=yt.title,output_path=file)

        # caption.save_captions("captions.txt")


while True:
    c = getInt("下载类型:1-> 音视频,2->字幕,9->退出:") 
    if c == 1:
        downloadVideo()
    elif c == 2:
        downCaption()
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
