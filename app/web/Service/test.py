from sanic import Sanic
import services.tasks.devCapImg as custom
from co6co_task.service import CustomTask, CuntomCronTrigger

ccc = CuntomCronTrigger.resolvecron("0 0 12 * * 2 *")
print(ccc)

app = Sanic()
app.prepare(debug=False)

subclasses = CustomTask.get_all_subclasses()
for subclass in subclasses:
    cl: custom.ICustomTask = subclass
    print(cl.name)
cupture = custom.DeviceCuptureImage()
cupture.capture_dev_image('rtsp://admin:lanbo12345@192.168.3.1:554/Streaming/Channels/1', R'D:\tmp\楼梯间.jpg')
