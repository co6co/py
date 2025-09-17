from selenium import webdriver
from selenium.webdriver.common.by import By

# 1. 启动 ChromeDriver 并打开 Chrome
# ChromeDriver 版本与 Chrome 浏览器版本严格匹配
#  https://sites.google.com/chromium.org/driver/ 下载相关版本
driver = webdriver.Chrome()  # 自动关联本地的 ChromeDriver

# 2. 发送指令：打开目标网页（如某网站登录页）
driver.get("http://127.0.0.3/system/")

# 3. 发送指令：输入用户名和密码
driver.find_element(By.CLASS_NAME, "el-input__inner").send_keys("admin")
driver.find_element(By.CLASS_NAME, "el-input__inner").send_keys("test_pwd")

# 4. 发送指令：点击“登录”按钮
driver.find_element(By.XPATH, "//button[@type='submit']").click()

# 5. 验证结果：判断是否登录成功（如是否出现“欢迎”文本）
if "欢迎" in driver.page_source:
    print("登录测试通过！")
else:
    print("登录测试失败！")

# 6. 关闭浏览器和 ChromeDriver
driver.quit()