协程相关代码：
https://www.cnblogs.com/chetianjian/p/17715303.html

sqlAlchemy：
https://docs.sqlalchemy.org/en/20/core/pooling.html

# python 最新版本
```
co6co==0.0.8
co6co.sanic-ext==0.0.4
co6co.db-ext==0.0.8
co6co.web-db==0.0.11
co6co.permissions==0.0.10

```

# ui 最新版本
```
co6co-ui==0.1.10 
co6co-right==0.0.8

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