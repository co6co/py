from co6co.utils import log

log.end_mark("动态代码")
log.start_mark("一、使用eval方式执行代码:", num=26)
result = eval('336//36')
print(result)

log.start_mark("二、使用exec方式执行代码:", num=26)
code = """
x = 5
y = 10
print("The sum of x and y is", x + y)
"""
print("\n执行代码:")
exec(code)


log.start_mark("三、使用预编译方式", num=26)
print(
    """
    # 预编译代码以提高性能
    ## 1. 定义了一个函数，
    ## 2. 编译这个源代码字符串，
    ## 3. 并使用 exec() 来执行它。
    ## 4. 调用函数 hello
    """
)
source = """
def hello(name='world'):
    print(f"Hello, {name}!")
def test(x,y):
    return x*y
"""
# 直接执行
exec(source)
hello('Python')

eval("hello('Python')")
# 编译执行
compiled_code = compile(source, "<string>", "exec")
exec(compiled_code)
hello('Python')
print(eval("test(10, 55)"))
eval("hello('Python')")
