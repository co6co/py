from asyncio import streams
import os
import re, base64
from win32crypt import CryptUnprotectData  ,CryptUnprotectData

import sqlite3

import win32con
import win32security

'''
def enable_privilege(privilege_name):
    """启用指定特权，返回是否成功"""
    # 打开令牌，需要 ADJUST_PRIVILEGES 权限
    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY,
    )

    # 查特权的 LUID
    luid = win32security.LookupPrivilegeValue(None, privilege_name)

    # 构造新的特权数组：(LUID, 属性)
    new_privs = [(luid, win32con.SE_PRIVILEGE_ENABLED)]

    # 调整令牌特权
    win32security.AdjustTokenPrivileges(token, False, new_privs)
    return win32api.GetLastError() == 0


privileges_to_enable = [
    win32security.SE_DEBUG_NAME,  # 调试程序
    win32security.SE_BACKUP_NAME,  # 备份文件
    win32security.SE_RESTORE_NAME,  # 还原文件
    win32security.SE_TAKE_OWNERSHIP_NAME,  # 取得所有权
    win32security.SE_LOAD_DRIVER_NAME,  # 加载驱动
    win32security.SE_INCREASE_QUOTA_NAME,  # 调整内存配额
    win32security.SE_IMPERSONATE_NAME,  # 模拟客户端
    win32security.SE_CREATE_GLOBAL_NAME,  # 创建全局对象
    win32security.SE_TCB_NAME,  # 以操作系统方式操作
]

enabled = []
failed = []

#for priv in privileges_to_enable:
#    success, msg = enable_privilege(priv)
#    if success:
#        enabled.append(priv)
#    else:
#        failed.append((priv, msg))
#
#print(len(failed) == 0, enabled, failed)
'''
# pip install keyring   # 可以调用 DPAPI 接口将密码存在本地
# import keyring
# keyring.set_password("nats", "myname", "password")  # 存入系统凭据
# pw = keyring.get_password("nats", "myname")          # 读取


EDGE_APP_DATA = "Microsoft\\Edge\\User Data"
STATE_FILE = "Local State"
LOGIN_DATA = "Default\\Login Data"
#print(os.getenv("LOCALAPPDATA"))
local_app_data = os.getenv("LOCALAPPDATA")
edge_path = os.path.join(local_app_data, EDGE_APP_DATA)
state_path = os.path.join(edge_path, STATE_FILE)


def get_orgin_keys(localState: str):
    key = re.findall(r'"encrypted_key":"(.*?)"', localState)[0]
    key20 = re.findall(r'"app_bound_encrypted_key":"(.*?)"', localState)[0]
    return {"v10": key, "v20": key20}

def decrypt_data_optimized(buffer: bytes,masterKeyV10,masterKeyV20) -> bytes:
    """
    优化版本：减少切片操作，提高性能
    """
    if not buffer or len(buffer) < 3:
        print("buffer too short or empty" )
        return None 
    try:
        # 检查版本前缀（不需要完整解码为字符串）
        if buffer.startswith(b"v10") or buffer.startswith(b"v11") or buffer.startswith(b"v20"):
            # 选择主密钥
            is_v20 = buffer.startswith(b"v20")
            master_key = masterKeyV20 if is_v20 else masterKeyV10
            if master_key is None or len(buffer) < 15:
                print("master_key is None or len(buffer) < 15" )
                return None
            
            # 提取各部分（避免多次切片）
            iv = buffer[3:15]           # 12字节 IV
            payload = buffer[15:]       # 剩余部分
            
            if len(payload) < 16:
                print("payload too short" )
                return None
            
            # 分离数据和认证标签
            data = payload[:-16]        # 密文数据
            tag = payload[-16:]         # 认证标签
            
            if not data:
                print("data is None" )
                return None
            
            # AES-GCM 解密
            try:
                aesgcm = AESGCM(master_key)
                decrypted = aesgcm.decrypt(iv, data, tag)
                return decrypted[32:] if is_v20 else decrypted
            except Exception as e:
                print("AES-GCM decryption failed" ,e)
                return None
                
        else:
            # DPAPI 解密
            try:
                decrypted_data, _ = CryptUnprotectData(
                    buffer, None, None, None, 0
                )
                return decrypted_data
            except Exception:
                return None
                
    except Exception as ex:
        print("decrypt_data_optimized error 2" ,ex)
        return None

def decrypt_password(data, key):
    print("ciphertext", data, len(data))
    nonce = data[3:15]
    
    print("iv", nonce, len(nonce))
    ciphertext = data[15:]
    print("data", ciphertext, len(ciphertext))
    if len(ciphertext) < 16:
        return None
    tag = ciphertext[-16:]
    data = ciphertext[:-16]
    print("tag", tag, len(tag))
    print("data", data, len(data)) 

    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, data,tag )

def decrypt_data(buffer: bytes, keys: bytes):
    key = base64.b64decode(keys["v10"])
    key20 = base64.b64decode(keys["v20"])
    Key10_type = key[0:5]
    #print("Key10_type", Key10_type)
    protected_data = key[5:]
    v10name, v10_key = CryptUnprotectData(protected_data, None, None, None, 0)
    #print("v10 name", v10name)
    #print("v10 key", v10_key)

    #Key20_type = key20[0:4]
    #print("Key20_type", Key20_type)
    #protected_data = key20[4:]
    #print("v20 key data", protected_data)
    #enable_privilege("SE_DEBUG_NAME")
    #data = CryptUnprotectData(protected_data, None, None, None, 0)
    #data = CryptUnprotectData(data, None, None, None, 0)
    #print("v20 data", data)
    #decrypt_password(buffer, v10_key)
    decrypted_data = decrypt_data_optimized(buffer, v10_key, None)
    print("decrypted_data", decrypted_data)


if __name__ == "__main__":
    keys = {}
    with open(state_path, "r") as f:
        data = f.read()
        keys = get_orgin_keys(data)

    state_path = os.path.join(edge_path, LOGIN_DATA)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    

    with sqlite3.connect(state_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT  action_url, username_value,password_value,password_type FROM logins"
        )
        # columns = [desc[0] for desc in cursor.description]
        # print("字段名:", columns)
        for row in cursor.fetchall():
            print("-"*90)
            print(row[2])
            decrypt_data(row[2], keys)
            print("="*90)

            # password = decrypt_password(row[2],keys)
            # print(password)
             
