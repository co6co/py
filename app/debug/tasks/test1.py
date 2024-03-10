import asyncio
from threading import Thread
from sanic import Sanic
from co6co.utils import log
import requests,os


class DemoTest( ):
    app:Sanic
    def __init__(self,app:Sanic) -> None:
        self.app=app
        pass
    def checkService(self):
        try:  
            response=requests.get("http://127.0.0.1:8084/v1/api/test",timeout=3)  
            if response.status_code==200:return True
            return False
        except Exception as e:
            print("error:",e)
            return False
    def run(self):
        try: 
            #app.m.terminate() # 关闭整个应用及其所有的进程  
            isRuning=self. checkService() 
            if  not isRuning: 
                log.info(">>>> 服务未能提供服务，即将重启 alarm...") 
                result=os.system('systemctl stop alarm.service && systemctl start alarm.service')
                log.warn(f">>>> 重启结果alarm...{result}")  
            else:
                log.info("8084 service is runing.") 
            #app.m.name.restart("","") # 重启特点的 worker  
        except Exception as e:
            log.err(f"檢測任務失敗。{e}")  
            
async def monitor(app:Sanic): 
    while True: 
        try: 
            await asyncio.sleep(60)	
            demo=DemoTest(app)
            Thread(target=demo.run ).start()
        except Exception as e:
            log.warn(f"执行任务失败：{e}")