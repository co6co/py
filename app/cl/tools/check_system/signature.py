import win32api
import win32security
import win32com.client
import os

def verify_file_signature(file_path):
    """
    验证文件数字签名
    返回: (bool, str) -> (是否有效, 签名者信息)
    """
    if not os.path.exists(file_path):
        return False, "文件不存在"

    try:
        # 方法1: 使用 Win32API (底层)
        # 注意: 这是简化的调用，完整实现需要处理复杂的结构体
        win32api.GetFileVersionInfo(file_path, '\\VarFileInfo\\Translation')
        
        # 方法2: 使用 Wintrust (更推荐，代码更简洁)
        # 调用 wintrust.dll 中的 WinVerifyTrust
        wtd = win32com.client.Dispatch("WinVerifyTrust.WinVerifyTrust")
        
        # 使用 WinVerifyTrust 验证
        # 返回值 0 表示成功 (TRUST_E_PROVIDER_UNKNOWN 等表示失败)
        result = win32api.WinVerifyTrust(
            None,  # HWND
            win32com.client.CLSIDFromString("{00AAC56B-CD44-11D0-8CC2-00C04FC295EE}"),  # Action ID (Generic Verify)
            file_path  # File path
        )
        
        if result == 0:
            # 获取签名者信息
            cert_info = win32security.CryptQueryObject(
                win32security.CERT_QUERY_OBJECT_FILE,
                file_path,
                win32security.CERT_QUERY_CONTENT_FLAG_ALL,
                win32security.CERT_QUERY_FORMAT_FLAG_ALL,
                None
            )
            # 提取证书上下文
            cert_ctx = cert_info[3]
            subject = win32security.CertGetNameString(
                cert_ctx,
                win32security.CERT_NAME_SIMPLE_DISPLAY_TYPE,
                0,
                None
            )
            return True, f"签名有效，签发给: {subject}"
        else:
            return False, f"签名无效，错误码: {result}"

    except Exception as e:
        return False, f"验证失败: {str(e)}"
if __name__ == "__main__":
    # 示例：验证 svchost.exe
    path = r"C:\Windows\System32\svchost.exe"
    is_valid, msg = verify_file_signature(path)
    print(f"路径: {path}")
    print(f"结果: {msg}")
