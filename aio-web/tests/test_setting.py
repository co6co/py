from co6co_db_ext.db_session import db_service,connectSetting
from co6co.data import DictNamespace
import pytest 
 

def test_connect_setting():
    config = connectSetting()
   
    dict=DictNamespace(**{"DB_HOST": "localhost", "port": 3306, "user": "root", "password": "123456", "db": "test"})
   
    for k in config.keys():
        print(k,dict.keys())
        if k in dict.keys():
            config[k] = dict[k]
        #assert config[k] == dict[k]
 
    print(config)
