
import multiprocessing
import sys,time

# 全局执行，子进程也能跑到，打包exe必备
multiprocessing.freeze_support()

def worker(arg):
    """子进程实际执行的任务"""
    for i in range(100):
        print(f"子进程运行: {arg}_{i}")
        time.sleep(0.1)
    return f"结果: {arg}"


def main():
    """主进程逻辑"""
    print("主进程开始")

    # 创建子进程
    p = multiprocessing.Process(target=worker, args=("test",)) 
    p.start()
    p.join() # 进程结束后，继续执行后面的代码

    #使用进程池模式
    with multiprocessing.Pool(2) as pool: # 除主进程又开两个进程
        res = pool.map(worker, ["work01","work02","work03","work04"])
        print(res)
    print("主进程结束")

if __name__ == '__main__':
    # 统一设置启动方式，避免平台差异导致奇怪问题
    try:
        # Windows 只能 spawn；Linux/mac 也兼容
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # 防止重复设置报错
        pass

    main()