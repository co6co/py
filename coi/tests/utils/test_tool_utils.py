import unittest
from co6co.utils.tool_util import *
from unittest.mock import Mock 


class Test_tool_util(unittest.TestCase):
    def test_list_to_tree(self):
        # Setup
        data=[
            {"id":0,"pid":None,},
            {"id":1,"pid":0,},
            {"id":2,"pid":0,},
            {"id":3,"pid":1,},
            {"id":4,"pid":2,},
            {"id":5,"pid":4,},
        ]
        #config = Mock(wx_config=config)
        
        
        # Action     
        result=list_to_tree(data,0,"pid","id") 
        # Assert
        self.assertEqual(len(result), len(result)) 
