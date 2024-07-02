from model.pos.wx import WxUserPO
class wxUser: 
    ownedAppid:str # 所属公众号
    openid:str 
    nickname:str
    sex:str
    language:str
    city:str
    province:str
    country:str
    headimgurl:str
    privilege:str 

    def __init__(self) -> None:
        self.ownedAppid=None # 所属公众号
        self.openid=None 
        self.nickname=None
        self.sex=None
        self.language=None
        self.city=None
        self.province=None
        self.country=None
        self.headimgurl=None
        self.privilege=None 
        pass


    def to(self, po:WxUserPO): 
        po.ownedAppid=self.ownedAppid if self.ownedAppid !=None else po.ownedAppid
        po.openid=self.openid if self.openid !=None else po.openid
        po.nickName=self.nickname if self.nickname !=None else po.nickName
        po.sex=self.sex if self.sex !=None else po.sex
        po.language=self.language if self.language !=None else po.language
        po.city=self.city if self.city !=None else po.city
        po.province=self.province if self.province !=None else po.province
        po.country=self.country if self.country !=None else po.country
        po.headimgUrl=self.headimgurl if self.headimgurl !=None else po.headimgUrl
        po.privilege=self.privilege if self.privilege !=None else po.privilege
        return po
    def __repr__(self) -> str:
        return f"class.{wxUser.__name__}>>openid:{self.openid},ownedAppid:{self.ownedAppid}"


