import string


class Point:
    def __init__(self, x, y, dist):
        self.x = x
        self.y = y
        self.dist = dist


# 创建一个Point实例的列表
points = [Point(i, i * 2, i * 3) for i in range(5)]

# 使用列表推导式风格修改dist值
# 注意这不是真正的列表推导式，而是一个简单的for循环
newDistList = [new_dist := point.dist + 10 for point in points]
print("新距离：", newDistList)

# 现在我们需要用新的值更新原始列表
# 没太多用 使用更加复杂，还是简单就可以
for point, new_dist in zip(points, [point.dist + 10 for point in points]):
    point.dist = new_dist

# 定义两个列表
names = ['Alice', 'Bob', 'Charlie', "a", 'b']
ages = [25, 30, 35, 123]
combined = list(zip(names, ages))
print(names, ages, '----zip--to_List-->', combined)
# 使用 zip() 和 dict() 创建字典
age_dict = dict(zip(names, ages))
print(names, ages, '----zip--to_dict-->', age_dict)

s1 = [123, 456]
s2 = [7, 8, 9, 10, 11]
s3 = s1+s2
print("list: s1->{} + s2->{}=s3->{}".format(s1, s2, s3))
print("s1 id:", id(s1), "\ts2 id:", id(s2), "\ts3 id:", id(s3))

print(string.digits, string.ascii_letters)
chrs = [chr(i) for i in range(33, 127)]
for i, a in enumerate(chrs):
    print(a, "->", str(ord(a)).ljust(3),  end="\t")
    if (i+1) % 9 == 0:
        print()
