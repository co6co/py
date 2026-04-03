from typing import Type, TypeVar, Generic

T = TypeVar("T")
K = TypeVar('K')

# Pytest 在收集测试时，将 TestB类误认为是测试类（因为它以 "Test" 开头）
# 定义类名是不能以 Test 开头


class ModelBase(Generic[T]):
    def __init__(self, a: T, b: T):
        self.a = a
        self.b = b

    def test_a(self):
        assert 1 == 1

    def get_a(self):
        return self.a


tt = ModelBase[int]
print(type(tt))
# tt.get_a() 不能调用 get_a

a = ModelBase(1, 2)
a.get_a()


class ModelBaseWithC(ModelBase[T], Generic[T, K]):
    """
    # 第一个类型参数 T：来自 TestA
    # 第二个类型参数 K：TestB 自己新增的
    # 总共：T 和 K 两个类型参数
    """

    def __init__(self, a: T, b: T, C: K):
        self.C = C
        super().__init__(a, b)

    def test_C(self):
        return self.C


test = ModelBaseWithC(1, 2, 'a')  # test=TestB[int,str](1,2,'a')
print(test.test_C(), type(test.test_C()))
print(test.C, type(test.C))
