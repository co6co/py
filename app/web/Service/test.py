import datetime
from services.croniter import croniter
# 当前时间
base_date = datetime.datetime.now()

# cron 表达式
# 秒、分、小时、日期、月份、周几（有些系统支持）、年份（可选）
cron_exp = "0 0 * * MON-FRI"

# 创建 croniter 对象
cron = croniter(cron_exp, base_date)

# 获取下一次满足条件的时间
next_time = cron.get_next(datetime.datetime)
print("Next scheduled time:", next_time)
