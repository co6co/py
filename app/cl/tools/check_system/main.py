from file import scan_process_paths
from signature import verify_file_signature
from process import scan_memory_for_injection
from dll import check_dll_hijacking
import psutil
def main():
    '''
    使用了
    pip install pywin32 psutil pefile cryptography

    '''
    print("=== Windows 安全巡检工具 ===")
    print("警告：请以管理员身份运行！\n")

    # 1. 扫描进程路径
    print("[*] 扫描进程路径欺骗...")
    suspicious_paths = scan_process_paths()
    for p in suspicious_paths:
        print(f"  [!] PID:{p['pid']} {p['name']} @ {p['path']}")
        print(f"      Reason: {p['reason']}")
        # 顺便验证签名
        valid, msg = verify_file_signature(p['path'])
        print(f"      Signature: {msg}")

    # 2. 深度扫描（针对系统关键进程）
    print("\n[*] 深度扫描关键进程 (svchost, lsass, explorer)...")
    critical_processes = ['svchost.exe', 'lsass.exe', 'explorer.exe']
    
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['name'].lower() in critical_processes:
                pid = proc.info['pid']
                print(f"\n[+] 分析进程: {proc.info['name']} (PID: {pid})")
                
                # DLL 劫持检查
                print("  - 检查 DLL 劫持...")
                dlls = check_dll_hijacking(pid)
                if dlls:
                    for d in dlls:
                        print(f"    [!] 可疑DLL: {d['path']}")
                
                # 内存注入检查
                print("  - 检查内存注入...")
                mem_regions = scan_memory_for_injection(pid)
                if mem_regions:
                    for m in mem_regions:
                        print(f"    [!] 可疑内存: {m['address']} Size:{m['size']} Note:{m['note']}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

if __name__ == "__main__":
    main()