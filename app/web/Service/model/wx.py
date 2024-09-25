from model.pos.wx import WxUserPO
from model.enum.wx import wx_authon_scope
from co6co.utils import getRandomStr
import time


class pageAuthonParam:
    """
    微信菜单认证 参数对象
    """
    code: str
    """
    微信服务器传的参数
    """
    appid: str
    backurl: str
    """
    snsapi_base 或者 snsapi_userinfo
    """
    state: str
    """
    微信服务透传过来的
    """
    @property
    def scope(self):
        """
        根据 state 是否包含 snsapi_base 来返回
        默认:snsapi_userinfo
        """
        if "snsapi_base" in self.state:
            return wx_authon_scope.snsapi_base
        return wx_authon_scope.snsapi_userinfo

    def __init__(self, appid: str) -> None:
        self.backurl = "/"
        self.appid = appid

    def __repr__(self) -> str:
        return f"class<Authon_param> ==>code:{self.code},appid:{self.appid},url:{self.backurl},scope:{self.scope},state:{self.state}"


class snsApiData:
    """
    {'access_token': '', 'expires_in': 7200, 'refresh_token': '', 'openid': 'oOQzt6UghEQ1VtNZHXjcqPiAD39E', 'scope': 'snsapi_base'}
    """
    access_token: str
    refresh_token: str
    openid: str
    expires_in: int
    scope: wx_authon_scope


class wxUser:
    ownedAppid: str  # 所属公众号
    openid: str
    nickname: str
    sex: str
    language: str
    city: str
    province: str
    country: str
    headimgurl: str
    privilege: str

    def __init__(self, ownedAppid: str = None) -> None:
        self.ownedAppid = ownedAppid  # 所属公众号
        self.openid = None
        self.nickname = None
        self.sex = None
        self.language = None
        self.city = None
        self.province = None
        self.country = None
        self.headimgurl = None
        self.privilege = None
        pass

    def to(self, po: WxUserPO):
        """
        临时对象转实体对象
        """
        po.ownedAppid = self.ownedAppid or po.ownedAppid
        po.openid = self.openid or po.openid
        po.nickName = self.nickname or po.nickName
        po.sex = self.sex or po.sex
        po.language = self.language or po.language
        po.city = self.city or po.city
        po.province = self.province or po.province
        po.country = self.country or po.country
        po.headimgUrl = self.headimgurl or po.headimgUrl
        po.privilege = self.privilege or po.privilege
        return po

    def __repr__(self) -> str:
        return f"class.{wxUser.__name__}>>openid:{self.openid},ownedAppid:{self.ownedAppid}"


class jsApiParam:
    """
    客户端申请jsapi 授权时传入的参数
    """
    url: str
    """
    http://xxxx.com/xxx.html?query=1
    不能有 # 后面的内容
    有参数 ?query=1

    """

    def __init__(self) -> None:
        self.url = None
        pass


class jsApiResult:
    """
    客户端申请jsapi 授权时 返回的结果
    """
    appId: str
    signature: str
    timestamp: str
    nonceStr: str

    def __init__(self) -> None:
        self.appId = None
        self.signature = None
        self.timestamp = str(int(time.time()))  # 十位 /秒级
        # self.timestamp = str(int(time.time()*1000))
        self.nonceStr = getRandomStr(5)
        pass

    def __str__(self) -> str:
        return "appid:{}    timestamp:{}    nonceStr:{}    signature{}".format(self.appId, self.timestamp, self.nonceStr, self.signature)
