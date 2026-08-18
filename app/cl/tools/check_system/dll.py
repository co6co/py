# DLL 劫持检测 (DLL Hijacking)

import pefile  # 用于解析 PE 文件
import win32process
import win32api

from file import is_suspicious_path 
from signature import verify_file_signature
def check_dll_hijacking(pid):
    """
    检查特定进程是否存在DLL劫持风险
    原理：检查加载的DLL是否位于System32，是否有有效签名
    """
    suspicious_dlls = []
    try:
        # 枚举进程模块
        h_process = win32api.OpenProcess(
            win32process.PROCESS_QUERY_INFORMATION | win32process.PROCESS_VM_READ,
            False,
            pid,
        )
        modules = win32process.EnumProcessModules(h_process)

        for mod in modules:
            try:
                mod_path = win32process.GetModuleFileNameEx(h_process, mod)

                # 检查路径
                is_suspicious, reason = is_suspicious_path(mod_path)

                # 检查签名
                is_valid, sig_msg = verify_file_signature(mod_path)

                # 如果路径可疑或签名无效，标记为可疑
                # 注意：某些合法的第三方DLL可能无签名，需要结合白名单判断
                if is_suspicious or not is_valid:
                    # 排除已知的无签名系统DLL（如果有）
                    suspicious_dlls.append(
                        {
                            "path": mod_path,
                            "path_reason": reason if is_suspicious else "正常",
                            "signature": sig_msg,
                        }
                    )
            except Exception:
                continue

    except Exception as e:
        # print(f"无法访问进程 {pid}: {e}")
        pass

    return suspicious_dlls
