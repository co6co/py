from typing import List


class associationParam:
    """
    关联参数
    """ 
    add: List[int]
    remove: List[int]
    def __init__(self, add: List[int] = [], remove: List[int] = []):
        self.add = add
        self.remove = remove
