from co6co.utils import log
 
def get_upload_path(appConfig)->str|None:
    """
    获取上传路径
    """ 
    if "biz_config" in appConfig and "upload_path" in appConfig.get("biz_config"):
        root=appConfig.get("biz_config").get("upload_path")
        return root
    log.warn("未配置上传路径：请在配置中配置[biz_config-->upload_path]")
    return None