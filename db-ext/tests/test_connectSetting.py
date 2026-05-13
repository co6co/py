from co6co_db_ext.db_session import connectSetting, DictNamespace


def test_connectSetting():
    init = connectSetting()
    assert len(init.keys()) == 0
    dict_data = {
        "DB_NAME": "test",
    }
    data = DictNamespace(**dict_data)
    config = connectSetting.create_default(data)
    for k in config.keys():
        print(k, config[k])
    assert config["DB_NAME"] == dict_data["DB_NAME"]
