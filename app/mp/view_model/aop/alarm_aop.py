from functools import wraps
from co6co.utils import log
from wechatpy import WeChatClient
from sanic import Sanic
from sanic.request import Request
from sanic.response import redirect,raw
from model import WechatConfig
from model.pos.biz import bizAlarmPO ,bizAlarmTypePO
from view_model.wx_config_utils import get_wx_configs, get_wx_config,crate_wx_cliet_by_config
from services.bll.wx_user import wx_user_bll
from services.bll.aralm import alarm_bll
import asyncio
from threading import Thread
from time import sleep, ctime
from utils.threadLoop  import ThreadEventLoop
 

def Alarm_Save_Succ_AOP(func):
    @wraps(func)
    def warpper(request:Request ,po:bizAlarmPO):
        allConfig:list[WechatConfig]=get_wx_configs(request)  
        loop = asyncio.get_event_loop()
        log.warn(f"主{id(loop)}") 
        for c in allConfig:  
            thread_tts = Thread(target=startAlarmPush,name=f"alarm_thread" ,args=(c,request.app,po))
            thread_tts.start()   
        return func(request,po)
    return warpper

def startAlarmPush(config: WechatConfig,app:Sanic,po:bizAlarmPO):
    # 通过查询订阅该公众号的用户
    try:  
        sleep(1)
        log.warn("任务... ")  
        t=ThreadEventLoop()    
        bll=wx_user_bll(app,t.loop)
        # 1. asyncio.run 在 没有正在运行的事件循环 的情况下运行协程的
        #wx_user_dict:list[dict]=asyncio.run(bll.get_subscribe_alarm_user(config.appid)) 
        #alarm= alarm_bll(app)
        #po:bizAlarmTypePO=asyncio.run(alarm.get_alram_type_desc(po.alarmType))
        # 2. 正在运行的事件循环
        # This event loop is already running   
        # import nest_asyncio 
        #loop = asyncio.get_event_loop()
        #wx_user_dict:list[dict]=loop.run_until_complete(bll.get_subscribe_alarm_user(config.appid))

        # 3. 创建任务
        #task=asyncio.create_task(bll.get_subscribe_alarm_user(config.appid))
        #wx_user_dict:list[dict]=asyncio.run(task) 
        # 4.底层使用
        #asyncio.ensure_future(coro())
        # 5. 
       
        wx_user_dict:list[dict]=t.runTask(bll.get_subscribe_alarm_user,config.appid) 
        alarm= alarm_bll(app,t.loop)
        print(po.alarmType)
        typePO=t.runTask( alarm.get_alram_type_desc,po.alarmType) 
        # 推送  
        if wx_user_dict!=None and len( wx_user_dict)>0 and typePO!=None:
            client=crate_wx_cliet_by_config(config)  
            
            msg=f'发现违反规则：{typePO.get("alarmType") }->{typePO.get("desc")}' 
            log.warn(f"通告信息。{config.name}。 。{msg},{client.access_token}") 
            log.err(wx_user_dict)
            for u in  wx_user_dict:   
                log.info(f"yshu Id:{u.get('appid')}，{u.get('appid')},type{type(u)}")
                #client.message.send_text(u.get("appid"),msg)
    except Exception  as e:
        log.err(f"告警失败：{e}")
 


