# -*- coding: utf-8 -*-
import unittest

from co6co.utils import hash
from co6co.utils import File
from co6co.utils import log

class TestFile(unittest.TestCase):
    content:str=None
    def setUp(self):
        self.content="abdd你好"
        pass 
    def test_base64(self): 
        data=hash.enbase64(self.content)
        self.assertEqual(data , 'YWJkZOS9oOWlvQ==') 
        self.assertEqual(hash.debase64(data),self.content)
        
    def test_md5(self):
        self.assertEqual(hash.md5(self.content),"64a593481db481a36ae0c753c521acac") 
        
    def tearDown(self):
        pass
    
def suite():
    #若测试用例都是以test开头，则用 makeSuite()方法一次性运行
    return unittest.makeSuite(TestFile )
    suite=unittest.TestSuite() 
    suite.addTest(TestFile("test_base64"))  #添加测试用例
    return suite

if __name__=="__main__":
    #文件使用方案1. python v_utils_test.py
    #文件使用方案2 python -m unittest v_utils_test
    # 使用方案1
    unittest.main(defaultTest='suite') 
    # 使用方案2
    #runner=unittest.TextTestRunner()
    #runner.run(suite2)
    