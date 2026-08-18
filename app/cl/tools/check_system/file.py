#路径欺骗检测 (Path Spoofing)

import psutil
import os

# 定义系统核心路径白名单
SYSTEM_WHITELIST_PATHS = [
    os.path.normcase(r"C:\Windows\System32"),
    os.path.normcase(r"C:\Windows\SysWOW64"),
    os.path.normcase(r"C:\Windows\WinSxS"),
    os.path.normcase(r"C:\Program Files"),
    os.path.normcase(r"C:\Program Files (x86)")
]

def is_suspicious_path(proc_path):
    """检查路径是否可疑"""
    if not proc_path:
        return True, "路径为空"
    
    norm_path = os.path.normcase(proc_path)
    
    # 检查是否在白名单中
    in_whitelist = any(norm_path.startswith(p) for p in SYSTEM_WHITELIST_PATHS)
    
    # 检查 Temp, AppData 等高危目录
    suspicious_keywords = [r'\temp', r'\appdata', r'\downloads', r'\users\public']
    in_suspicious = any(keyword in norm_path for keyword in suspicious_keywords)
    
    if not in_whitelist and in_suspicious:
        return True, f"高危路径: {proc_path}"
    if not in_whitelist:
        return True, f"非标准系统路径: {proc_path}"
        
    return False, "路径正常"

def scan_process_paths():
    """扫描所有进程路径"""
    suspicious_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['exe']:
                is_suspicious, reason = is_suspicious_path(proc.info['exe'])
                if is_suspicious:
                    suspicious_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'path': proc.info['exe'],
                        'reason': reason
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return suspicious_processes