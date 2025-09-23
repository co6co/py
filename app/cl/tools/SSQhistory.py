# 书写双色球中奖结果
# 要求
# 1. 通过时间来获取指定时间的中奖结果
# 2. 通过网络来获取最新的中奖结果
# 3. 将交过保存在 一个.md 文件
# 4. 红球以,隔开, 蓝球以+隔开

# pip install selenium 安装selenium库
# - 需要下载ChromeDriver，并确保其版本与本地Chrome浏览器版本匹配
# - ChromeDriver需要放在系统PATH中，或者在代码中指定其路径
import json
import datetime
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

class SSQResult:
    def __init__(self, issue:str, date:str, red_balls:list[str], blue_ball:str):
        self.issue = issue
        self.date = date
        self.red_balls = red_balls
        self.blue_ball = blue_ball
    def __str__(self):
        return f"{self.issue} {self.date} {','.join(self.red_balls)} + {self.blue_ball}"

class SSQHistory:
    def __init__(self):
        # 使用新的链接 - 中国福彩网阳光开奖页面
        self.base_url = "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/"
        # 结果保存的目录
        self.save_dir = "./ssq_results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _get_driver(self):
        """创建并配置Selenium WebDriver"""
        chrome_options = Options()
        # 无头模式，不显示浏览器窗口
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # 创建WebDriver实例
        driver = webdriver.Chrome(options=chrome_options)
        return driver 
    def get_all_pages(self,onlyFirst=False):
        """获取所有页面的URL"""
        driver = None
        try: 
            all_results = [] 
            driver = self._get_driver() 
            driver.get(self.base_url)  
            WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'table'))
                )  
            page_source = driver.page_source 
            results, has_next_page = self.get_results_current_page(page_source) 
            if results:
                all_results.extend(results) 
            
            while has_next_page and not onlyFirst:
                # 直接使用Selenium查找下一页按钮并点击 
                next_page_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.layui-laypage-next:not(.layui-disabled)'))
                ) 
                next_page_button.click() 
                # 等待下一页加载完成
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'table'))
                ) 
                page_source = driver.page_source
                # 删除重复的这一行
                # page_source = driver.page_source
                results, has_next_page = self.get_results_current_page(page_source)
                if results:
                    all_results.extend(results) 
            return all_results
        except Exception as e: 
            print(f"获取所有页面失败: {e}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    print(f"关闭浏览器失败: {e}")
    # 修改函数名中的拼写错误
    def get_results_current_page(self, page_source: str):
        """获取页面上所有期号的结果"""
        try: 
            # 解析HTML内容
            soup = BeautifulSoup(page_source, 'html.parser')  
            # 检查是否还有下一页
            has_next_page = False
            pagination = soup.find('a', {'class': 'layui-laypage-next'})
            if pagination and 'layui-disabled' not in pagination.get('class', []): 
                has_next_page = True
            # 查找开奖结果表格
            table = soup.find('table', {'class': 'ssq_table'})
            if not table:
                # 如果找不到ssq_table类的表格，尝试找普通表格
                table = soup.find('table')
            all_results = []
            if table:
                # 获取表格中的所有行
                rows = table.find_all('tr')
                
                # 遍历所有行，跳过表头
                for row in rows[1:]:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # 提取期号和日期
                        issue = cells[0].text.strip()
                        date_text = cells[1].text.strip()
                        date = date_text.split('(')[0] if '(' in date_text else date_text
                        
                        # 提取开奖号码 
                        red_balls = []
                        for redDiv in cells[2].find_all("div", {"class": "qiu-item-wqgg-zjhm-red"}): 
                            red_balls.append(redDiv.text.strip())
                        
                        # 添加错误处理，防止找不到蓝球时程序崩溃
                        blue_ball_div = cells[2].find("div", {"class": "qiu-item-wqgg-zjhm-blue"})
                        if blue_ball_div:
                            blue_ball = blue_ball_div.text.strip()
                            result = SSQResult(issue, date, red_balls, blue_ball)
                            all_results.append(result) 
        
            return all_results, has_next_page  # 始终返回列表，即使为空
        except Exception as e:
            print(f"获取当前页结果失败: {e}") 
            return [], False  # 出错时返回空列表和False
    
    def save_to_md(self, result: SSQResult | list[SSQResult], filename: str = None):
        """将结果保存到markdown文件"""
        try:
            # 确保结果是一个列表
            results = [result] if isinstance(result, SSQResult) else result
            
            # 确定文件名
            if not filename:
                filename = "ssq.md"
            
            # 保存文件
            file_path = os.path.join(self.save_dir, filename)
            
            # 检查文件是否存在，避免重复写入相同内容
            existing_results = set()
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            existing_results.add(line.strip())
            
            # 对结果进行排序（按期号从大到小排序）
            results_to_write = []
            for item in results:
                result_str = str(item)
                if result_str not in existing_results:
                    results_to_write.append(result_str)
            
            # 按期号排序（假设期号在字符串的开头）
            allList=list(results_to_write)
            allList.extend(existing_results)
            allList.sort(key=lambda x: x.split()[0], reverse=True)
            
            # 只写入新的结果
            with open(file_path, 'w', encoding='utf-8') as f:
                for result_str in allList:
                    f.write(result_str + "\n")
            
            print(f"结果已保存至: {file_path}")
            return True
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False

     

# 测试代码
if __name__ == "__main__":
    try:
        ssq = SSQHistory() 
        # 获取页面上所有期号的结果
        isFirst=input("是否仅获取第一页,否获取全部数据？(y/n)")
        print("\n正在获取页面结果...")
        
        all_results = ssq.get_all_pages(onlyFirst=isFirst.startswith('y'))
        if all_results:
            print(f"共找到{len(all_results)}期结果")
            saved = ssq.save_to_md(all_results)
            if not saved:
                print("保存结果失败")
        else:
            print("未能获取到所有结果")
    except Exception as e:
        print(f"程序执行出错: {e}")
     