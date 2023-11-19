from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from threading import Thread
import time
from co6co.utils import log


def times():
    print(time.strftime("定时任务执行了: %Y-%m-%d-%H_%M_%S", time.localtime()))

# 删除存放30天的文件
def param_job(arg:str, beforeDay="30"):
    """
    :param dir_path: 文件路径
    :param beforeDay: 需要删除的天数
    :return:
    """
    log.info(f"参数 {arg} : {beforeDay}") 
    
dir_path=R"C:\temp"
sched = BackgroundScheduler()
sched.add_job(times, CronTrigger.from_crontab('*/1 * * * *'))
sched.add_job(param_job, CronTrigger.from_crontab('*/2 * * * *'), args=(dir_path, "30"))
sched.add_listener()

t = Thread(target=sched.start,daemon=True)
t.start()
t.join()
#sched.start()