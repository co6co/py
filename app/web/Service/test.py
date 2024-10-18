from services.tasks import CuntomCronTrigger, Scheduler
from datetime import datetime
import time
# startTimeTask()
# startIntervalTask()
x = CuntomCronTrigger.resolvecron("0 0 12 5/2 * *")
print("下载执行时间：", x.get_next_fire_time(previous_fire_time=datetime.now(), now=datetime.now()))


code = """
import datetime
def main():
    print(datetime.datetime.now(), "123456789")

"""
res, e = Scheduler.parseCode(code)
print(res, e)
if res:
    s = Scheduler()
    s.addTask(code, "1/15 * * * * *")

time.sleep(30)
