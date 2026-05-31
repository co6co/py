class TestDemo:
     
    def get_2(i: int | str):
        """
        2. 转换为字典
        """
        if isinstance(i, int):
            return str(i)
        return i
    def b(self):
        v=self.get_2(1)
        print(v) 