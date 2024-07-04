from exec_command import execute_command, parse_netstat_output
from sendMail import send_email
from queryIP import get_location_by_ip
import os
# 要执行的命令
command = 'netstat -ano | findstr :3389 | findstr "ESTABLISHED"'
# 执行命令
result = execute_command(command)
ips = parse_netstat_output(result, 2)

if ips == None:
    username = os.getlogin()
    print ("本地登录",username)
    exit(0)
desc = ""
for ip in ips:
    parts = ip.split(":")
    city, region, country, lat, lon = get_location_by_ip(parts[0])
    desc = f'{desc}\r\n{ip}->{country}.{region}.{city},{lat, lon}'
domail = 'xxxx.com'
send_email(
    to_email=f"admin@{domail}",
    from_email=f"alarm@{domail}",
    smtp_server=f"smtp.{domail}",
    smtp_port=25,
    smtp_username=f"alarm@{domail}",
    smtp_password="113BD93DA244FDD14E88B975F3B2A014",
    subject="用户登录通知",
    message=f'''
        用户已成功登录Windows系统。
        来自：{desc}
        '''
)
