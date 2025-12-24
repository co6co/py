
import matplotlib 
def init_matplotlib_params():
    """
    初始化 matplotlib 字体，用于显示中文字符
    """
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文字符
    matplotlib.rcParams['axes.unicode_minus'] = False   # 用于显示负号，避免出现乱码