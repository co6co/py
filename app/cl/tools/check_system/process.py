# 注入检测 (Process Injection)
import ctypes
from ctypes import wintypes
import psutil

# 定义必要的 Windows API 结构和常量
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
MEM_COMMIT = 0x1000
MEM_PRIVATE = 0x20000
PAGE_EXECUTE_READWRITE = 0x40

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", wintypes.LPVOID),
        ("AllocationBase", wintypes.LPVOID),
        ("AllocationProtect", wintypes.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
    ]


def scan_memory_for_injection(pid):
    """
    扫描进程内存，查找具有执行权限的私有内存区域 (RX/RWX)
    这是检测反射型DLL注入、Shellcode注入的常用方法
    """
    suspicious_memory = []
    try:
        h_process = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if not h_process:
            return []

        mbi = MEMORY_BASIC_INFORMATION()
        address = 0

        while ctypes.windll.kernel32.VirtualQueryEx(
            h_process, address, ctypes.byref(mbi), ctypes.sizeof(mbi)
        ):
            # 检测条件：内存已提交 + 是私有内存 + 具有执行权限
            # 注意：JIT编译器也会创建此类内存，需结合进程类型判断
            if (
                mbi.State == MEM_COMMIT
                and mbi.Type == MEM_PRIVATE
                and mbi.Protect in (PAGE_EXECUTE_READWRITE, 0x20, 0x10)
            ):  # PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE等
                # 尝试读取该内存区域，如果失败或包含异常特征则记录
                buffer = ctypes.create_string_buffer(mbi.RegionSize)
                bytes_read = wintypes.SIZE_T()

                if ctypes.windll.kernel32.ReadProcessMemory(
                    h_process, address, buffer, mbi.RegionSize, ctypes.byref(bytes_read)
                ):
                    # 简单的特征码匹配（示例：查找 MZ 头，可能藏有PE文件）
                    if buffer.raw.startswith(b"MZ"):
                        suspicious_memory.append(
                            {
                                "address": hex(address),
                                "size": mbi.RegionSize,
                                "protect": hex(mbi.Protect),
                                "note": "发现隐藏的PE文件头 (MZ)",
                            }
                        )

            address += mbi.RegionSize
            if address >= 0x7FFFFFFF:  # 32位边界
                break

        ctypes.windll.kernel32.CloseHandle(h_process)

    except Exception as e:
        pass

    return suspicious_memory
