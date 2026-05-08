import pytest
from co6co.data import DictNamespace


def test_dict_namespace():
    # 创建对象
    items = DictNamespace(name="Alice", age=25, city="Beijing")

    # 1. 属性访问
    assert items.name == "Alice"
    assert items.age == 25

    # 2. 字典访问
    assert items['name'] == "Alice"
    assert items.get('age') == 25

    # 3. 字典操作
    items['job'] = "Engineer"
    assert items.job == "Engineer"

    # 4. 字典方法
    assert list(items.keys()) == ['name', 'age', 'city', 'job']
    assert list(items.values()) == ['Alice', 25, 'Beijing', 'Engineer']
    assert list(items.items()) == [('name', 'Alice'), ('age', 25), ('city', 'Beijing'), ('job', 'Engineer')]

    # 5. 更新操作
    items.update({'salary': 100000, 'department': 'IT'})
    assert items.salary == 100000

    # 6. 删除操作
    job = items.pop('job')
    assert job == "Engineer"
    assert ('job' in items) == False

    # 7. 遍历
    for key, value in items.items():
        print(f"{key}: {value}")

    # 8. 转换为字典
    regular_dict = items.to_dict()
    assert regular_dict == {'name': 'Alice', 'age': 25, 'city': 'Beijing', 'salary': 100000, 'department': 'IT'}

    # 9. 从字典创建
    new_items = DictNamespace.from_dict({'x': 10, 'y': 20})
    assert new_items.x == 10
