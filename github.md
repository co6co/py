查詢最新的github 地址
https://sites.ipaddress.com/github.com/

## 1. 撤销git add .
方法1：
```
git reset # 撤销所有已 add 的文件从暂存区撤出，但保留工作区的修改，相当于撤销 git add .的效果
```
方法2：--staged表示只操作暂存区（stage），不影响工作目录。
```
git restore --staged .
```

## 2. 撤销：git commit -m "data"
### 方法1：
将当前分支的 HEAD指针回退到上一个提交（HEAD~1），但保留工作区和暂存区的修改（即本次提交的内容会回到暂存区，可直接修改后重新提交）。
```
git reset --soft HEAD~1
```

```
# 查看提交历史（确认要撤销的提交）
git log --oneline

# 撤销最近一次提交（保留修改区）
git reset --soft HEAD~1

# 此时 git status 会显示修改仍在暂存区，可重新编辑后提交
git status
```

###  方法2： git reset HEAD~1（默认 --mixed模式）
提交记录被删除，修改保留在工作区，需重新 git add后再提交。

回退 HEAD指针到上一个提交，并将暂存区的修改移回工作区（即本次提交的内容回到工作区，需重新 git add后提交）
```
# 撤销最近一次提交（修改回到工作区）
git reset HEAD~1  # 等价于 git reset --mixed HEAD~1

# 此时 git status 会显示修改在工作区，需重新 add
git status
```

###  方法3 慎用 彻底丢弃本次提交及修改（慎用！）
如果确定本次提交的内容完全错误，且不需要保留任何修改（例如误提交了敏感信息或大文件），可使用
` git reset --hard`
回退 HEAD指针到上一个提交，并强制覆盖工作区和暂存区（即本次提交的所有修改会被永久删除）。