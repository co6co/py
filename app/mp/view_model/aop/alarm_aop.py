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
from co6co.task.thread import ThreadEvent
import json , traceback
from co6co_sanic_ext.utils import JSON_util

def Alarm_Save_Succ_AOP(func):
    @wraps(func)
    def warpper(request:Request ,po:bizAlarmPO):
        try:
            allConfig:list[WechatConfig]=get_wx_configs(request)  
            #loop = asyncio.get_event_loop()
            #log.warn(f"主{id(loop)}") 
            for c in allConfig:  
                thread_tts = Thread(target=startAlarmPush,name=f"alarm_thread" ,args=(c,request.app,po))
                thread_tts.start() 
        except Exception as e:
            log.err("warpper告警失败：:{e}")   
        return func(request,po)
    return warpper

def startAlarmPush(config: WechatConfig,app:Sanic,po:bizAlarmPO):
    # 通过查询订阅该公众号的用户
    try:  
        sleep(0.5)
        log.warn("任务... ")  
        t=ThreadEvent()    
        alarm= alarm_bll(app,t.loop)
        wx_user_dict:list[dict]=t.runTask(alarm.get_subscribe_alarm_user,config.appid) 
        if wx_user_dict==None or len( wx_user_dict)==0 :
            log.warn("需要告警的用户为0,不推送告警。")
            return 
        typePO=t.runTask( alarm.get_alram_type_desc,po.alarmType) 
        # 推送  
        if typePO!=None:
            log.info(f"需发送用户数：{len( wx_user_dict)}")
            client=crate_wx_cliet_by_config(config)   
            alarmType=typePO.get("alarmType")
            alarmDesc=typePO.get("desc")
            templatId=None
            data=None
            msg=None
            if config.alarm_tamplate_id==None:
                msg=f'发现违反规则：{ alarmType}->{alarmDesc}'  
            else: 
                templatId=config.alarm_tamplate_id
                data={
                    "alarmType":{"value":alarmType},
                    "alarmDesc":{"value":alarmDesc},
                    "alarmTime":{"value":po.alarmTime}  
                } 
                #date 无法使用默认序列化   
                data=JSON_util().encode(data) 
                data=json.loads(data) 
            for u in  wx_user_dict:
                openid=u.get("openid")
                nickName=u.get("nickName")
                log.warn(f'发送 告警消息 {templatId or msg}\t to \t{openid}{nickName}')
                if templatId==None: sendMessage(client,openid,msg,nickName)
                else: sendTemplateMessage(client,openid,nickName,templatId,data)
    except Exception  as e: 
        log.err(f"告警失败：{e},{traceback.format_exc()}")

def sendMessage(client:WeChatClient, openId, msg:str ,nickName:str):
    """
    发送普通消息
    """
    try: 
        # 45015
        # send_text 用户长期为与公众号联系将不能发送消息
        jsonData=client.message.send_text(openId,msg) 
        log.warn(f"发送文本消息<<：{jsonData}" )
    except Exception as e:
        log.err(f"发送文本消息-->{openId}{nickName}失败：{e}")

def sendTemplateMessage(client:WeChatClient, openId,nickName:str, tempId, data ,url:str=None):
    """
    发送模板消息
    """
    try:  
        jsonData=client.message.send_template(openId,tempId,data=data,url=url)
        log.warn(f"发送模板消息<<：{jsonData}" )
    except Exception as e:
        log.err(f"发送模板消息-->{openId}{nickName}失败：{e}")

 


