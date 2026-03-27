from services.tasks.devCapImg import getRtspAddress,cap_image 
from model.enum import DeviceCategory, DeviceVender
#import capsys
import pytest # pip install pytest
import os

import cv2

@pytest.fixture #提供测试所需的预设数据、环境或资源
def data():
    return [0,"192.168.1.100","admin","admin2023",DeviceVender.Hikvision.key]
def test_getRtspAddress(data):
    data[0]=DeviceCategory.monitor
    data[4]=DeviceVender.Hikvision.key
    result = getRtspAddress(*data)
    print("海康监控：",result)
    assert len(result) ==1

    data[0]=DeviceCategory.ParkAndPass
    data[4]=DeviceVender.Hikvision.key
    result = getRtspAddress(*data)
    print("海康一体机：",result)
    assert len(result) ==2

    data[0]=DeviceCategory.monitor
    data[4]=DeviceVender.Dahua.key
    result = getRtspAddress(*data)
    print("大话监控：",result)
    assert len(result) ==1
    
    data[0]=DeviceCategory.monitor
    data[4]=DeviceVender.Uniview.key
    result = getRtspAddress(*data)
    print("宇视监控：",result)
    assert len(result) ==1

    data[0]=DeviceCategory.monitor
    data[4]=DeviceVender.TPLink.key
    result = getRtspAddress(*data)
    print("TP-link监控：",result)
    assert len(result) ==1

    data[0]=DeviceCategory.monitor
    data[4]='unknown'
    result = getRtspAddress(*data)
    print("未知厂家监控：",result)
    assert len(result) ==0

    data[0]=DeviceCategory.monitor
    data[4]=''
    result = getRtspAddress(*data)
    print("空厂商[使用默认海康监控]：",result)
    assert len(result) ==1

    data[0]=DeviceCategory.monitor
    data[4]=None
    result = getRtspAddress(*data)
    print("None厂商[使用默认海康监控]：",result)
    assert len(result) ==1
    # 捕获标准输出
    #captured = capsys.readouterr()
    #captured.out

@pytest.fixture  
def videoUrls()->list:
    """
    需要单独的数据，需要设备网络的支持
    """ 
    ip="192.168.1.64"
    pwd="admin123"
    port=554
    '''
    data= [DeviceCategory.monitor,ip,"admin",pwd,DeviceVender.Hikvision.key,port]
    data= [DeviceCategory.monitor,ip,"admin",pwd,DeviceVender.TPLink.key,port]   
    data= [DeviceCategory.monitor,ip,"admin",pwd,DeviceVender.Uniview.key,port]
    '''
    data= [DeviceCategory.monitor,ip,"admin",pwd,DeviceVender.Dahua.key,port]
    return getRtspAddress(*data)

def test_image(videoUrls ): 
    output_image_path = "./test.jpg" 
    for v in videoUrls: 
        print(v)
        result = cap_image(v,output_image_path) 
        print(result) 
        assert result[0] == True
        assert result[1] == 0 
        if os.path.exists(output_image_path):
            os.remove(output_image_path)


