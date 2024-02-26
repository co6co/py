import re

def to_camelcase(name: str) -> str:
    """下划线转驼峰(小驼峰)"""
    return re.sub(r'(_[a-z])', lambda x: x.group(1)[1].upper(), name)
def to_underscore(name: str) -> str:
    """驼峰转下划线"""
    if '_' not in name:
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    else:
        raise ValueError(f'{name}字符中包含下划线，无法转换')
    return name.lower()

def list_to_tree( data_list:list, root:any, pid_field:str, id_field:str):
    """
    list 转 tree
    //todo 视乎 [] 有什么内在的联系
    """ 
    resp_list = [i for i in data_list if i.get(pid_field) == root]  
    for i in data_list:
        i['children'] = [j for j in data_list if i.get(id_field) == j.get(pid_field)] 
    return resp_list 