查詢最新的github 地址
https://sites.ipaddress.com/github.com/

撤销
git add .
方法1：
git reset # 撤销所有已 add 的文件从暂存区撤出，但保留工作区的修改，相当于撤销 git add .的效果
方法2：--staged表示只操作暂存区（stage），不影响工作目录。
```
git restore --staged .
```