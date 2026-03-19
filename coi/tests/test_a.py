from typing import Type,TypeVar,Generic
from ..co6co.enums import Base_EC_Enum
T=TypeVar("T")
K=TypeVar('K')
class TestA(Generic[T]):
    def __init__(self,a:T,b:T):
        self.a=a
        self.b=b
    def test_a(self):
        assert 1 == 1
    def get_a(self):
        return self.a 

tt=TestA[int]
print(type(tt))
#tt.get_a() 不能调用 get_a  

a=TestA(1,2) 
a.get_a()

class TestB(TestA[T],Generic[T,K]):
    """
    # 第一个类型参数 T：来自 TestA
    # 第二个类型参数 K：TestB 自己新增的
    # 总共：T 和 K 两个类型参数
    """
    def __init__(self,a:T,b:T,C:K):
        self.C=C
        super().__init__(a,b)
    def test_C(self):
        return self.C

test=TestB(1,2,'a') # test=TestB[int,str](1,2,'a')
print(test.test_C(),type(test.test_C()))
print(test.C,type(test.C))
   
   
 