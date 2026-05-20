from co6co_db_ext.jwt_service import JwtService
import pytest


@pytest.fixture
def jwt_service():
    return JwtService("123456")

def test_create_token(jwt_service:JwtService):
    data1={"userOpenId": "123456"}
    data2={"userOpenId": "123456"}
    data3={"role": "admin","remark":"管理员"}
    time=6000
    k=2
    result = jwt_service.create_token(data1, data2,data3=data3,expire_seconds=time,k=k)
    print(result)

    assert result["accessToken"] is not None
    assert result["refreshToken"] is not None
    accessToken=result["accessToken"]
    refreshToken=result["refreshToken"] 

    tokens=[accessToken,refreshToken]
    for index,token in enumerate( tokens): 
        assert token is not None  
        print("++++++++",token)
        print(token["token"])
        tokenStr=token["token"]
        decoded_token:dict = jwt_service.decode(tokenStr) 
        assert decoded_token.get("data") is not None 
        if index==0:
            assert decoded_token.get("data")==data1
        else:
            assert decoded_token.get("data")==data2

  
    assert result['data3']==data3 
