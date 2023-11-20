# nohup 
```
nohup python app.py >> /home/cy/logs/app.log 2>&1 &

说明：
no hang up 的缩写，运行一个进程，账号/终端关闭退出不结束
&: 表示and的符号
&: 指定后台运行
0: stdin (standard input)
1: stdout (standard output)
2: stderr (standard error)

2>&1: 标准错误重定向到标准输出

nohup python app.py > /home/cy/logs/app.log 2>error.txt # 错误文件会输出到error.txt中

```

# systemctl脚本
```
[Unit]

Description=audit service
After =network.target,syslog.target

[Service]

ExecStart=/home/cy/work/start.sh
User=root
Group=root

[Install]

WantedBy=multi-user.target
```
## 检查
```
systemctl status your-service-name.service
systemctl status your-dependency-service-name.service
journalctl -u your-service-name.service
```

## 设置
```
systemctl enable test.service # 开机启动
systemctl disable test.service # 取消开机启动

```

