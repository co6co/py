
# -*- coding: utf-8 -*-
# 动态调用python代码

'''
## 1. eval(expression, globals=None, locals=None)->Any|None

## 只能是单个运算表达式 (注意eval不支持任意形式的赋值操作)，而不能是复杂的代码逻辑
#globals : 变量作用域，全局命名空间，如果被提供，则必须是一个字典对象
#locals  : 变量作用域，局部命名空间，如果被提供，可以是任何映射对象


## 2. exec(object[, globals[, locals]])-->None
object: 必选参数, 表示需要被指定的Python代码。它必须是字符串或code对象。如果object是一个字符串,该字符串会先被解析为一组Python语句, 然后在执行(除非发生语法错误)。如果object是一个code对象,那么它只是被简单的执行。
globals:可选参数, 表示全局命名空间(存放全局变量)，如果被提供，则必须是一个字典对象。
locals: 可选参数, 表示当前局部命名空间(存放局部变量),如果被提供,可以是任何映射对象。如果该参数被忽略,那么它将会取与globals相同的值。


区别:
eval()函数只能计算单个表达式的值,而exec()函数可以动态运行代码段。

## 3. compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)
1、    source:是一串字符串的源码,或者是AST(抽像语法树)对象数组,就是需要执行的代码对象。
2、    filename:参数filename用于在执行代码报错的运行时错误消息中显示该参数对应的信息,当source是执行代码从文件中读取的代码字符串时,则可以存放文件名,如果不是从文件里读取源码来编译,那么这里可以放一些用来标识这些代码的字符串,其值理论上是任何字符串,没有特殊要求,一般都放'<string>',用于表示前面的source是个字符串,如果source放AST,则可以标识为'<AST>';
3、    mode:三个取值,分别是'exec'、'single' 、'eval',如果是'exec'表示编译的是一段代码或模块, 'single'表示编译的是一个单独的语句, 'eval'表示编译的是一个表达式而不是一个语句。
这三种模式中,老猿初步验证凡是'single'模式能编译的就能'exec'模式编译,'eval'和二者不能互换。
4、    flags和dont_inherit
这两个参数是组合使用,可选参数 flags 和 dont_inherit 控制在编译 source 时要用到哪个 future 语句。
 如果两者都未提供(或都为零)则会使用调用 compile() 的代码中有效的 future 语句来编译代码。 如果给出了 flags 参数但没有 dont_inherit (或是为零) 则 flags 参数所指定的 以及那些无论如何都有效的 future 语句会被使用。 如果 dont_inherit 为一个非零整数,则只使用 flags 参数 -- 在调用外围有效的 future 语句将被忽略。
future 语句使用比特位来指定,多个语句可以通过按位或来指定。具体特性的比特位可以通过 __future__ 模块中的 _Feature 类的实例的 compiler_flag 属性来获得。
不知道各位有明白的没有,以上这段解释直接来自于Python 标准库,老猿只是照抄,没有看懂,估计涉及Python的高级特性future,以后再研究吧,我们暂时都用缺省值。
5、    optimize:optimize到Python的代码优化机制。
Python为了适应不同的执行要求定义了几种代码优化的策略:
1)    缺省值是-1,表示使用命令行参数-O中获取的优化等级为准;
2)    如果设置值为0,是没有优化,__debug__为true支持debug信息(if __debug__语句下的语句,就是开发者根据需要加入的调试信息)在运行中展示;
3)    如果设置值为1,assert语句被删除,__debug__设置为false确保调试语句不执行;
4)    如果设置值为2,除了设置值为1的功能之外,还会把代码里文档字符串也删除掉,达到最佳优化结果。

compile函数返回结果    
1、    如果编译通过,结果可以生成字节码(类型code)或者AST(抽像语法树),字节码可以使用函数exec()或eval来执行,而AST可以使用eval()来继续编译(关于AST的内容本节都不介绍,ATS 对象:Abstract Syntax Tree,抽象语法树,是源代码语法结构的一种抽象表示。关于抽象语法树大家可以参考:https://zhuanlan.zhihu.com/p/26988179;
2、    exec 语句:exec 执行储存在字符串或文件中的Python语句,相比于 eval,exec可以执行更复杂的 Python 代码。需要说明的是在 Python2 中exec不是函数,而是一个内置语句;
3、    如果编译的源码不合法,此函数会触发 SyntaxError 异常;如果源码包含 空字节(空字符串),则3.5版本以前会触发 ValueError 异常,3.5版本后则不会触发可以编译通过并执行。注意:
1)    在 'single' 或 'eval' 模式编译多行代码字符串(这些串必须是一个完整语句或表达式而不是多个语句或表达式)时,输入必须以至少一个换行符结尾;
2)    如果编译足够大或者足够复杂的字符串成 AST 对象时,Python 解释器会因为 Python AST 编译器的栈深度限制而崩溃
'''

expr = """
count = 10
for index in range(count):
    print(index)
"""


def print_str():
    eval("print('hello world')")


def add(x, y):
    """
    执行带返回值的表达式 
    """
    return eval("x+y")


def exec_eval():
    """
    出错
    expr 只能是单个表达式 
    """
    eval(expr)


def exec_for():
    exec(expr)


def eval_fun():
    """
    eval调用方法
    """
    eval("print_str()")


def exec_fun():
    """
    exec调用方法
    """
    print(__name__)
    exec("print_str()")


def exec_file(file_name, func_name):
    """
    file_name: 文件名
    func_name: 执行的方法
    """
    with open(file_name, "rb") as f:
        source_code = f.read()
    exec_code = compile(source_code, file_name, "exec")
    scope = {}
    exec(exec_code, scope)
    f = scope.get(func_name, None)
    f()


if __name__ == "__main__":
    print_str()
    print(add(12, 56))
    exec_for()
    print("调用文件方法:")
    exec_file(__file__, "exec_fun")
    print("------------------------------------------")
    exp = compile('select_max(a , b)', '', 'eval')

    def select_max(x, y):
        return x if x > y else y
    c = eval('select_max(3 , 5)', {"__builtins__": {}}, {'select_max': select_max})
    print("变量:", c)

    for i in range(10):
        a = i
        b = i + 10
        c = eval(exp)
        print("c is {}".format(c))
