from co6co.utils.modules import deprecated


def test_fun():
    @deprecated("use new_api() instead")
    def old_func():
        pass

    old_func()
def test_cls():
    @deprecated("use new_api() instead")
    class OldCls:
        pass
    c=OldCls()
    print(c)

