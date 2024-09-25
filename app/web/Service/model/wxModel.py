from __future__ import annotations


class wxCacheData:
    """
    微信登录需要保存的数据
    """
    userId: int
    openId: str
    ownAppid: str
    nickName: str
    headUrl: str
    accountId: str

    def __str__(self) -> str:
        return "<{}>userId:{},openId:{},nickName:{},ownAppid:{},accountId:{}".format(wxCacheData.__name__, self.userId, self.openId, self.nickName, self.ownAppid, self.accountId)

    def __init__(self) -> None:
        self.userId = None
        self.openId = None
        self.ownAppid = None
        self.nickName = None
        self.headUrl = None
        self.accountId = None

    def check(self, oldCacheData: wxCacheData):
        """
        如果当前为空使用老值
        """
        if oldCacheData != None:
            # 其他信息认为比较重要必传
            self.nickName = self.nickName or oldCacheData.nickName
            self.headUrl = self.headUrl or oldCacheData.headUrl
