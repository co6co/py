from pytubefix import YouTube
from pytubefix.cli import on_progress

proxys = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
while True:
    c = input("输入URL,q:quit:")
    if c== 'q':
        break
    url = c
    yt = YouTube(url,proxies=proxys, on_progress_callback=on_progress)
    print(yt.title)

    ys = yt.streams.get_audio_only()
    ys.download()
    print("FFF")
    