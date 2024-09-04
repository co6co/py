from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix import Playlist

url = input("输入要下载的URL:")
url = "https://www.youtube.com/watch?v=lxOFGvHBsTY"
proxys = {"http": "http://127.0.0.1:9667", "https": "http://127.0.0.1:9667"}
# yt = YouTube(url, on_progress_callback=on_progress, proxies=proxys)
yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, on_progress_callback=on_progress)
print("标题：", yt.title)


def downloadVideo():
    """
    下载视频或音频
    """
    list = yt.streams.all()

    def checkItem():
        """
        选择要下载的序号
        """
        for item in list:
            print(item)
        itag = input("input itag:")
        if itag == "q":
            return None
        return int(itag)
    while True:
        itag = checkItem()
        if itag == None:
            break
        checkedList = [i for i in list if i.itag == itag]

        for checked in checkedList:
            checked.download()


def downCaption():
    caption = yt.captions.all()
    print(*caption)
    # caption = yt.captions.get_by_language_code('en')
    # caption.save_captions("captions.txt")


while True:
    c = input("下载类型:1-> 音视频,2->字幕,9->退出")
    c = int(c)
    if c == 1:
        downloadVideo()
    elif c == 2:
        downCaption()
    else:
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
