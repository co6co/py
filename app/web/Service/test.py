from co6co_web_db.view_model import BaseMethodView
from typing import Callable
import services.tasks.custom as custom


subclasses = custom.get_all_subclasses()
for subclass in subclasses:
    cl: custom.ICustomTask = subclass
    print(cl.name)
cupture = custom.DeviceCuptureImage()
cupture.capture_dev_image('rtsp://admin:lanbo12345@192.168.3.1:554/Streaming/Channels/1', R'D:\tmp\楼梯间.jpg')
