# visualize_results.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def create_visualizations(benchmarks, compression_results, concurrency_results):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('管道传输性能分析', fontsize=16, fontweight='bold')
    
    # 1. 数据大小 vs 传输速度
    if benchmarks:
        ax1 = axes[0, 0]
        sizes = [b["data_size_mb"] for b in benchmarks]
        send_speeds = [b["send_speed_mbps"] for b in benchmarks]
        recv_speeds = [b["recv_speed_mbps"] for b in benchmarks]
        
        ax1.plot(sizes, send_speeds, 'o-', label='发送速度', linewidth=2, markersize=8)
        ax1.plot(sizes, recv_speeds, 's-', label='接收速度', linewidth=2, markersize=8)
        ax1.set_xlabel('数据大小 (MB)', fontsize=12)
        ax1.set_ylabel('传输速度 (MB/s)', fontsize=12)
        ax1.set_title('数据大小 vs 传输速度', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 压缩效果对比
    if compression_results:
        ax2 = axes[0, 1]
        patterns = list(set([r["pattern"] for r in compression_results]))
        sizes = list(set([r["size_mb"] for r in compression_results]))
        
        # 按模式分组
        pattern_data = {}
        for pattern in patterns:
            pattern_data[pattern] = [
                r["compression_ratio"] for r in compression_results 
                if r["pattern"] == pattern
            ]
        
        x = np.arange(len(sizes))
        width = 0.2
        
        for i, (pattern, ratios) in enumerate(pattern_data.items()):
            offset = width * i
            ax2.bar(x + offset, ratios, width, label=pattern)
        
        ax2.set_xlabel('数据大小 (MB)', fontsize=12)
        ax2.set_ylabel('压缩比', fontsize=12)
        ax2.set_title('不同数据模式的压缩效果', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width * (len(patterns) - 1) / 2)
        ax2.set_xticklabels([f"{s}MB" for s in sorted(sizes)])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 并发性能
    if concurrency_results:
        ax3 = axes[1, 0]
        concurrencies = [r["concurrency"] for r in concurrency_results]
        throughputs = [r["throughput"] for r in concurrency_results]
        success_rates = [r["success_rate"] for r in concurrency_results]
        
        # 双Y轴图表
        ax3_throughput = ax3
        ax3_success = ax3.twinx()
        
        line1 = ax3_throughput.plot(concurrencies, throughputs, 'o-', 
                                    color='tab:blue', linewidth=2, markersize=8, 
                                    label='吞吐量 (连接/秒)')
        line2 = ax3_success.plot(concurrencies, success_rates, 's-', 
                                 color='tab:red', linewidth=2, markersize=8, 
                                 label='成功率 (%)')
        
        ax3_throughput.set_xlabel('并发连接数', fontsize=12)
        ax3_throughput.set_ylabel('吞吐量 (连接/秒)', fontsize=12, color='tab:blue')
        ax3_success.set_ylabel('成功率 (%)', fontsize=12, color='tab:red')
        ax3_throughput.set_title('并发性能分析', fontsize=14, fontweight='bold')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3_throughput.legend(lines, labels, loc='upper left')
        
        ax3_throughput.grid(True, alpha=0.3)
    
    # 4. 时间分布
    if benchmarks:
        ax4 = axes[1, 1]
        sizes = [b["data_size_mb"] for b in benchmarks]
        send_times = [b["send_time"] for b in benchmarks]
        recv_times = [b["recv_time"] for b in benchmarks]
        total_times = [s + r for s, r in zip(send_times, recv_times)]
        
        width = 0.25
        x = np.arange(len(sizes))
        
        ax4.bar(x - width, send_times, width, label='发送时间', alpha=0.8)
        ax4.bar(x, recv_times, width, label='接收时间', alpha=0.8)
        ax4.bar(x + width, total_times, width, label='总时间', alpha=0.6)
        
        ax4.set_xlabel('数据大小 (MB)', fontsize=12)
        ax4.set_ylabel('时间 (秒)', fontsize=12)
        ax4.set_title('时间分布分析', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{s:.0f}" for s in sizes], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = f"performance_charts_{timestamp}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    
    print(f"图表已保存到: {chart_file}")
    plt.show()
    
    return chart_file

# 在 benchmark.py 的 main 函数中添加
if __name__ == "__main__":
    # ... 之前的测试代码 ...
    
    # 可视化结果
    chart_file = create_visualizations(benchmarks, compression_results, concurrency_results)
    print(f"可视化图表: {chart_file}")