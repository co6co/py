import win32serviceutil
import win32service
import win32api
import win32con

from co6co.utils.win import execute_command


def get_all_services():
    """获取系统中所有服务的信息"""
    services = []
    # 打开服务控制管理器
    hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ENUMERATE_SERVICE)

    try:
        # 枚举所有服务
        statuses = [
            win32service.SERVICE_ACTIVE,
            win32service.SERVICE_INACTIVE,
            win32service.SERVICE_STATE_ALL
        ]

        for status in statuses:
            # 枚举服务，每次最多获取256个
            service_info = win32service.EnumServicesStatus(hscm, win32service.SERVICE_WIN32, status)

            for (short_name, desc, status_code) in service_info:
                # 获取服务的详细配置
                try:
                    config = win32serviceutil.QueryServiceConfig(short_name)
                    start_type = config[4]  # 启动类型

                    # 转换启动类型为可读文本
                    start_type_text = {
                        win32service.SERVICE_AUTO_START: "自动",
                        win32service.SERVICE_DEMAND_START: "手动",
                        win32service.SERVICE_DISABLED: "禁用",
                        win32service.SERVICE_AUTO_START_DELAYED: "自动(延迟启动)"
                    }.get(start_type, f"未知({start_type})")

                    # 转换状态码为可读文本
                    status_text = {
                        win32service.SERVICE_STOPPED: "已停止",
                        win32service.SERVICE_START_PENDING: "启动中",
                        win32service.SERVICE_STOP_PENDING: "停止中",
                        win32service.SERVICE_RUNNING: "运行中",
                        win32service.SERVICE_CONTINUE_PENDING: "继续中",
                        win32service.SERVICE_PAUSE_PENDING: "暂停中",
                        win32service.SERVICE_PAUSED: "已暂停"
                    }.get(status_code, f"未知({status_code})")

                    services.append({
                        "短名称": short_name,
                        "描述": desc,
                        "状态": status_text,
                        "启动类型": start_type_text
                    })
                except Exception as e:
                    # 有些服务可能无法访问详细信息
                    services.append({
                        "短名称": short_name,
                        "描述": desc,
                        "状态": f"无法获取状态: {str(e)}",
                        "启动类型": "未知"
                    })

    finally:
        # 关闭服务控制管理器句柄
        win32service.CloseServiceHandle(hscm)

    # 去重并按服务名称排序
    unique_services = {s["短名称"]: s for s in services}.values()
    return sorted(unique_services, key=lambda x: x["短名称"])


if __name__ == "__main__":
    print("正在获取Windows系统服务信息...\n")
    services = get_all_services()

    # 打印服务信息，格式化输出
    print(f"共找到 {len(services)} 个服务:")
    print(f"{'短名称':<30} {'状态':<10} {'启动类型':<15} 描述")
    print("-" * 100)

    for service in services:
        print(f"{service['短名称']:<30} {service['状态']:<10} {service['启动类型']:<15} {service['描述']}")
