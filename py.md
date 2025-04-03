协程相关代码：
https://www.cnblogs.com/chetianjian/p/17715303.html

sqlAlchemy：
https://docs.sqlalchemy.org/en/20/core/pooling.html

pip install requests --index-url https://mirrors.aliyun.com/pypi/simple/
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
pip install -i https://pypi.org/simple/ some-package
pip install --upgrade --no-deps co6co --index-url https://pypi.org/simple/
# python 最新版本
```
co6co==0.0.26
co6co.web_session=0.0.1
co6co.sanic-ext==0.0.10
co6co.db-ext==0.0.13
co6co.web-db==0.0.14
co6co.permissions==0.0.27

```

```
pip install wechatpy==1.8.18
pip install cryptography
```
# ui 最新版本
npm install --registry=http://registry.npm.taobao.org/ isomorphic-streams
npm install --registry=https://registry.npmjs.org/ co6co

```
co6co-ui==0.1.25
co6co-right==0.0.33
co6co-wx==0.0.2



# 更新到满足 package.json 中定义的版本范围内的最新版本
npm update <package-name>
# 安装最新的版本（即使它超出了当前定义的版本范围）
npm install <package-name>@latest
# 升级指定斑斑
npm install <package-name>@<version>
```




## [Dlib](http://dlib.net/files/) 安装

dlib GPU 版本安装：
1、去官网 http://dlib.net/ 下载 dlib 压缩包，并解压；
2、安装 cmake，使用 pip install cmake 或 conda install cmake 安装即可；
3、去 dlip 解压目录下，执行 `python setup.py install`，等待完成安装。

判断dlib是否可以使用GPU：
```
import dlib
dlib.DLIB_USE_CUDA   # True 表示可以使用 GPU
```

# git proxy
```
git config --global https.proxy http://127.0.0.1:1080
git config --global https.proxy https://127.0.0.1:1080

git config --global --unset http.proxy
git config --global --unset https.proxy

npm config delete proxy
```