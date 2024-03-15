from sanic import Request


def getSiteId(request: Request):
    '''
    获取 json siteId
    '''
    return request.json.get("siteId")
