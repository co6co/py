from dataclasses import dataclass
@dataclass
class UserInfo:
    """
    微信用户信息
    {'openId': 'o7hOp7RV8Ocs38pmd3FYtn0MfvSw', 'nickName': '微信用户', 'gender': 0, 'language': '', 'city': '', 'province': '', 'country': '', 'avatarUrl': 'https://thirdwx.qlogo.cn/mmopen/vi_32/POgEwh4mIHO4nibH0KlMECNjjGxQUq24ZEaGT4poC6icRiccVGKSyXwibcPq4BWmiaIGuG1icwxaQX6grC9VemZoJ8rg/132', 'watermark': {'timestamp': 1779084450, 'appid': 'wx94c44488ec00ab8e'}}
    """
    openId: str
    nickName: str
    gender: int
    language: str
    city: str
    province: str
    country: str
    avatarUrl: str
    watermark: dict
       
