import unittest
from view_model.wx_config_utils import *
from unittest.mock import Mock
from utils import WechatConfig


class TestWxConfig(unittest.TestCase):
    def test__success__get_wx_configs(self):
        # Setup
        config = [
            {
                "name": "公众1",
                "appid": "wxa96e7921ab04afab",
                "appSecret": "0e2fe5bded860a9d8466fab5d1680ef1",
                "token": "abc_cy",
                "encodingAESKey": "Rxsoy3HI275p1AcyHuw1VZXfAqy9meJ23kKNofF1xLI",
                "encrypt_mode": "normal"
            },
            {
                "name": "测试公众号",
                "appid": "wx181aa5d9ce286cf0",
                "appSecret": "0b19ad3a1be39f71028409c05351a43a",
                "token": "token123456",
                "encodingAESKey": "",
                "encrypt_mode": "normal"
            }
        ]
        #config = Mock(wx_config=config)
      
        request = Mock( ) 
        request.app=Mock( ) 
        request.app.config=Mock( ) 
        request.app.config.wx_config= config 
        print(request.app.config.wx_config) 
        # Action
        result:list[WechatConfig]=get_wx_configs(request)
        # Assert
        self.assertEqual(len(config), len(result))
        self.assertEqual(config[1].get("appid"), result[1].appid)
