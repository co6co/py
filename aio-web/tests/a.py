import sys
# 很奇怪的 version_info  
print(sys.version_info,type(sys.version_info))
if sys.version_info < (3, 9):
    raise RuntimeError("aiohttp 3.x requires Python 3.9+")
else:
    print(sys.version_info,dir(sys.version_info))
    version_info_type = type(sys.version_info)
    #new_version = sys.version_info._replace(minor=10)
    #print("old->",sys.version_info,'new->',new_version)
    #它的字段是固定的：major, minor, micro, releaselevel, serial。
    # 获取sys.version_info的类型
    VersionInfo = type(sys.version_info)

    # 创建新的实例
    #vi = VersionInfo(3, 7, 0, 'final', 0)
    new_version = VersionInfo._make([3, 11, 0, 'final', 0])
    print(new_version) 
    

