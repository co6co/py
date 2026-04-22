from model.apphelp import get_config

def test_get_config():
    config = get_config()
    print(config,type(config))
    print(config["rtsp_url"])
    assert isinstance(config , dict)
    assert config is not None
