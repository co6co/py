import webbrowser


class webClient:
    url: str = None

    def __init__(self, url: str) -> None:
        self.url = url
        pass

    def open(self):
        # webbrowser.open(self.url) 默认浏览器打开
        path = R"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(path))
        browser = webbrowser.get('chrome')
        browser.open(self.url)


if __name__ == "__main__":
    client = webClient(
        "https://blog.csdn.net/wangyuxiang946/article/details/132231956")
    client.open()
