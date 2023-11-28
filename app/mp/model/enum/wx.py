
from co6co.enums import Base_Enum

class wx_encrypt_mode(Base_Enum):
    normal="normal",0
    compatible="compatible",1
    safe="safed",2 


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

class wx_event_type(Base_Enum): 
    subscribe="subscribe",1
    unsubscribe="unsubscribe",2
    subscribe_scan="subscribe_scan",3 #用户扫描二维码关注事件
    scan="scan",4 #用户扫描二维码事件
    location="location",5 #location
    click="click",6 #点击菜单拉取消息事件
    view="view",7 # 点击菜单跳转链接事件
    masssendjobfinish="masssendjobfinish",8 # 群发消息任务完成事件
    templatesendjobfinish="templatesendjobfinish",9 # 模板消息任务完成事件
    scancode_push="scancode_push",10 #扫码推事件
    scancode_waitmsg="scancode_waitmsg",11 # 扫码推事件且弹出“消息接收中”提示框的事件
    pic_sysphoto="pic_sysphoto",12 # 弹出系统拍照发图的事件
    pic_photo_or_album="pic_photo_or_album",13 # 弹出拍照或者相册发图的事件
    pic_weixin="pic_weixin",14 # 弹出微信相册发图器的事件 
    location_select="location_select",15 #弹出地理位置选择器的事件
    card_pass_check="card_pass_check",20# 卡券审核事件推送
    card_not_pass_check="card_not_pass_check",21 
    user_get_card="user_get_card",22 #领取事件推送
    user_gifting_card="user_gifting_card",23 #转赠事件推送
    user_del_card="user_del_card",24 #卡券删除事件推送
    user_consume_card="user_consume_card",25 #卡券核销事件推送
    user_pay_from_pay_cell="user_pay_from_pay_cell",26 #卡券买单事件推送
    user_view_card="user_view_card",27 #进入会员卡事件推送
    user_enter_session_from_card="user_enter_session_from_card",28 #从卡券进入公众号会话事件推送
    update_member_card="update_member_card",29#会员卡内容更新事件
    card_sku_remind="card_sku_remind",30 #卡券库存报警事件
    card_pay_order="card_pay_order",31#券点流水详情事件
    submit_membercard_user_info="submit_membercard_user_info",40 #会员卡激活事件推送
    merchant_order="merchant_order",41
    kf_create_session="kf_create_session",42
    kf_close_session="kf_close_session",43
    kf_switch_session="kf_switch_session",44
    device_text="device_text",50
    device_bind="device_bind",51
    device_unbind="device_unbind",52
    device_subscribe_status="device_subscribe_status",53
    device_unsubscribe_status="device_unsubscribe_status",54
    shakearound_user_shake="shakearound_user_shake",55
    poi_check_notify="poi_check_notify",56
    wificconnected="wificconnected",57
    qualification_verify_success="qualification_verify_success",60 #资质认证成功事件
    qualification_verify_fail="qualification_verify_fail",61 #资质认证失败事件
    naming_verify_success="naming_verify_success",62 #名称认证成功事件
    naming_verify_fail="naming_verify_fail",63 #名称认证失败事件
    annual_renew="annual_renew",64 #年审通知事件
    verify_expired="verify_expired",65 # 认证过期失效通知
    user_scan_product="user_scan_product",70 # 打开商品主页事件
    user_scan_product_enter_session="user_scan_product_enter_session",71 #进入公众号事件
    user_scan_product_async="user_scan_product_async",72 # 地理位置信息异步推送事件
    user_scan_product_verify_action="user_scan_product_verify_action",80# 商品审核结果事件
    subscribe_scan_product="subscribe_scan_product",81 #用户在商品主页中关注公众号事件
    user_authorize_invoice="user_authorize_invoice",82 # 用户授权发票事件
    update_invoice_status="update_invoice_status",83 #发票状态更新事件
    submit_invoice_title="submit_invoice_title",84 #用户提交发票抬头事件
    user_enter_tempsession="user_enter_tempsession",90 #小程序用户进入客服消息
    view_miniprogram="view_miniprogram",91 #从菜单进入小程序事件
    wxa_media_check="wxa_media_check",92 #异步检测结果通知事件