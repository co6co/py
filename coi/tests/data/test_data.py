from co6co.data import fibonacci, take, primes_fast

#  T = TypeVar("T", int, str) # T 为 int 或者 str
#  T = TypeVar("T") # T 为任何类型
#  T = TypeVar("T", bound=A) # T 为A或A的子类
#  X = TypeVar("AType", bound=A) # 建议 X 和 AType 命名一致
'''
def process(obj: T) -> T:
    return obj
def create_one(self, cls:Type[T]):
    return cls()
class Container:
    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value
'''

def test_fibonacci(): 
    print(list(take(fibonacci(), 9)))
    assert list(take(fibonacci(), 10)) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
def test_primes_fast(): 
    print(list(take(primes_fast(), 10000)))
    assert list(take(primes_fast(), 10)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_gen_error_code():
    from co6co.data.result import gen_error_code
    x=gen_error_code('error')
    print(hex(abs( "error".__hash__())).replace("0x","") .upper() )
    print("error".__hash__())
    print(x) 
def test_xx():
    import hashlib

    s = "error"
    h = hashlib.md5(s.encode()).hexdigest()[6:15].upper()
    print(h)