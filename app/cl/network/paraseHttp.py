from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import time


def get_dynamic_text_content(url, category: int = 0):
    """
    :param url
    :param category:0:chrome,1:firefox

    包括执行 javascripts
    类似与浏览其的行为
    """
    try:
        # 设置 ChromeDriver 路径，需根据实际情况修改
        # https://sites.google.com/chromium.org/driver/downloads
        # chromedriver->chrome
        if category == 0:
            service = ChromeService('G:\\ToolExe\\chrome\\chromedriver\\chromedriver.exe')
            driver = webdriver.Chrome(service=service)
        else:
            # geckodriver->Firefox
            service = FirefoxService('G:\\ToolExe\\chrome\\chromedriver\\geckodriver.exe')
            firefox_options = Options()
            firefox_options.binary_location = 'F:\\Program Files\\Mozilla Firefox\\firefox.exe'
            driver = webdriver.Firefox(service=service, options=firefox_options)

        # 打开网页
        driver.get(url)

        # 可以根据需要调整等待时间，确保页面的 JavaScript 代码执行完毕
        time.sleep(5)

        # 获取页面源代码
        page_source = driver.page_source
        # 使用 BeautifulSoup 解析页面源代码
        soup = BeautifulSoup(page_source, 'html.parser')
        text = soup.get_text()
        return text
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭浏览器
        driver.quit()
    return None
