from co6co.enums import Base_Enum
class WechatConfig:
    """
    配置信息[开发信息| 服务器配置]
    开发信息:appid,appSecret
    服务器配置:url, token,encodingAESKey,encrypt_mode
    
    """
    appid:str=None,		                #    公众号的appid
    appSecret:str=None,
    
    token:str=None, 		            #    token
    encodingAESKey:str=None,		            # 公众号的secret
    encrypt_mode:str=None
    def __init__(self) -> None:
        pass


class wx_message_type(Base_Enum):
    """
    文本消息 MsgType=="text"
    图片消息 MsgType=="image"
    语音消息 MsgType=="voice"
    视频消息 MsgType=="video"
    地理位置消息 MsgType=="location"
    链接消息 MsgType=="link"
    事件消息 MsgType=="event"
    关注、取消事件 Event=="subscribe",Event=="unsubscribe"
    """
    text="text",0
    image="image",1
    voice="voice",2
    shortvideo="shortvideo",3
    video="video",4
    location="location",5
    link="link",6
    event="event",9
    miniprogrampage="miniprogrampage",10