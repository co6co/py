from co6co.enums import Base_EC_Enum
from typing import List,Dict 

class Task_Type (Base_EC_Enum):
    """
    任务类型
    """
    down_task= 'down_task','下载任务',0 
    setting_light= 'settting_light','设置灯光参数',1 
    

class Task_Statue (Base_EC_Enum):
    """
    任务状态
    """
    created= 'created','创建',0
    starting= 'starting','开始任务',1
    finshed= 'finshed','完成',2
    error= 'error','完成,有错',3
    canel= 'canel','任务取消',9 
    