import py


class ClassName:
    _name: str = "Class_变量"

    def __init__(self, name=None):
        self._name = name or self.__class__._name
    pass


def test_class_name():
    name = "变量"
    cc = ClassName(name)
    print("tt.__class__.__name__:", cc.__class__.__name__)
    assert cc.__class__.__name__ == "ClassName"
    print("cc._name:", cc._name)
    assert cc._name == name
