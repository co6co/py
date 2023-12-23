# 文件权限
```
# lsattr .user.ini
----i--------e-- .user.ini

#chattr -i .user.ini
# lsattr .user.ini
-------------e-- .user.ini

chown -R username /wwww
```

# 服务所在路径（Ubuntu/centos）
`/usr/lib/systemd/system/`