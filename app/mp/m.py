from xml.etree import ElementTree
# django web 框架
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from wechat_sdk import WechatConf
from wechat_sdk import WechatBasic
from wechat_sdk.messages import TextMessage, ImageMessage, VoiceMessage, VideoMessage
# from wechat_sdk.exceptions import ParseError
from wechat_sdk.messages import EventMessage


conf = WechatConf(
    token='xxx', 		#自定义的token
    appid='wxxxxxxxxxx',		# 公众号的appid
    appsecret='xxxxxxxxxxxxxxxxx',		# 公众号的secret
    encrypt_mode='normal',)
wechat = WechatBasic(conf=conf)

@csrf_exempt
def check(request):
    if request.method == "GET":		
    # 用于在微信基本配置的服务器配置中，响应微信发送的Token验证
        try:
            signature = request.GET.get("signature")
            if signature:
                timestamp = request.GET.get("timestamp")
                nonce = request.GET.get("nonce")
                echostr = request.GET.get("echostr")
                if wechat.check_signature(signature, timestamp, nonce):
                    return HttpResponse(echostr)
            return HttpResponse(u"微信验证失败")

        except Exception as e:
            return HttpResponse(e)
    else:
        menu = {											# 根据自己需求定义
                'button': [
                    {
                        "name": "服务",
                        "sub_button": [
                            {
                                "type": "view",
                                "name": "规划",
                                "url": "https://xxx.com"		
                            },
                        ],
                    },
                    {
                        "name": "分类",
                        "sub_button": [
                            {
                                "type": "view",
                                "name": "精选",
                                "url": "http://www.xxx.com"
                            },
                        ],

                    },
                    {
                        "name": "加入",
                        "sub_button": [
                            {
                                "type": "media_id",
                                "name": "图片",
                                "media_id": "xxxxxxxxxxx"		# 获取的素材的media_id
                            },
                        ]
                    },
                ]
                }
    try:
        # wechat.create_menu(menu)		# 创建菜单，创建完成可以注释掉，每次请求都创建会浪费接口调用次数
        wechat.parse_data(request.body)
        if isinstance(wechat.message, TextMessage):
            content = wechat.message.content
            print('content:', content)
            if wechat.message.content == "问卷":
                content = "参与 \n--> <a href='https://xxx.com/xxxt'>戳此进入</a>"
            else:
                content = "欢迎来到公众号^_^！"
            xml = wechat.response_text(content=content)
            return HttpResponse(xml, content_type="application/xml")
        elif isinstance(wechat.message, ImageMessage):
            picurl = wechat.message.picurl
            media_id = wechat.message.media_id
            xml = wechat.response_image(media_id=media_id)
            return HttpResponse(xml, content_type="application/xml")
        elif isinstance(wechat.message, VoiceMessage):
            media_id = wechat.message.media_id
            format = wechat.message.format
            recognition = wechat.message.recognition
            xml = wechat.response_voice(media_id=media_id)
            return HttpResponse(xml, content_type="application/xml")
        elif isinstance(wechat.message, VideoMessage):
            media_id = wechat.message.media_id
            xml = wechat.response_video(media_id=media_id)
            return HttpResponse(xml, content_type="application/xml")
        elif isinstance(wechat.message, EventMessage):
            if wechat.message.type == 'subscribe':  # 关注事件(包括普通关注事件和扫描二维码造成的关注事件)
                root = ElementTree.fromstring(request.body)
                print('root:', root)
                from_user = root.findtext(".//FromUserName")
                print('fromUser:', from_user)		# 此时可以获取到针对该公众号而言的用户的openid，自行添加逻辑即可
                xml = wechat.response_text(content="欢迎关注")
                return HttpResponse(xml, content_type="application/xml")

    except Exception as e:
        print(e)
    return HttpResponse(u"这是首页") 