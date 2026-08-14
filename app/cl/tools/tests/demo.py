from asyncio import streams
import os
import re, base64
import time
from win32crypt import CryptUnprotectData, CryptUnprotectData

import sqlite3

import win32con
import win32security
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.fernet import Fernet


from win32api import CopyFile

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
#EDGE_APP_DATA = "Chromium\\User Data"
STATE_FILE = "Local State"
LOGIN_DATA = "Default\\Login Data"
# print(os.getenv("LOCALAPPDATA"))
local_app_data = os.getenv("LOCALAPPDATA")
edge_path = os.path.join(local_app_data, EDGE_APP_DATA)
state_path = os.path.join(edge_path, STATE_FILE)


def get_orgin_keys(localState: str):
    key = re.findall(r'"encrypted_key":"(.*?)"', localState)[0]
    key20 = re.findall(r'"app_bound_encrypted_key":"(.*?)"', localState)[0]
    return {"v10": key, "v20": key20}
    
def decrypt_key(key: str,skip:int=5):
    """
    key 解密

    """
    data = base64.b64decode(key) 
    _, key = CryptUnprotectData(data[skip:], None, None, None, 0) 
    return key 
def aes_decrypt_data(key,nonce,ciphertext,tag):
    from cryptography.hazmat.primitives.ciphers import   algorithms, modes
    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce, tag),  # 把 tag 传进去，GCM 会自动验证完整性
    ).decryptor() 
    plaintext = decryptor.update(ciphertext) # + decryptor.finalize()
    return plaintext

def decrypt_v10_data(data, key): 
    nonce = data[3:15]  
    ciphertext = data[15:-16]
    tag = data[-16:] 
    plaintext = aes_decrypt_data(key,nonce,ciphertext,tag)
    return plaintext  
def decrypt_v20_data(data, key): 
    nonce = data[3:15]  
    ciphertext = data[15:-16]
    tag = data[-16:] 
    plaintext = aes_decrypt_data(key,nonce,ciphertext,tag)
    return plaintext  

 


def decrypt_data(data:bytes, keys):
    """
    解密数据
    """
    '''
    -----BEGIN PGP MESSAGE-----
Version: GnuPG v1

hQEMA4fLSGZcsDhlAQgArI+NKLT0ETqqZwIOAlnPppFD9vkbYG8CfDsOmpZWe3QP
GupI0e35VKcalFJl4up/QAAAwKrh1KmZKZowYvk0E7TxhKHmFrunqwWD1yMtxBV8
rDSmu11S3up8Ov1kdyxhlVSrytpp5EydLwMjqeocj9S6wRFqBtklBZBRBPZnIVU/
KRZPlYUvYDtHhhR7QdTR10twyiuQojQIKu6eJ8tzYVFXptX0Wp7aJRhdqAqQhzj2
F8Ya5+UnCvzV+x5pXovV0nwOmOEcY7s51NKjHyEDnrU9IeIcPoUT8lctZ6tgv5WC
A8Vth9OWX/anWu2ZcrXbqlxzIOSli1MJp3e3AQkbKNLBGQGNXI8sr7mVh4ULYW3b
xNbd5NqjVLs06VPpAqa+wyejZRxRW7Vu0pctO1QmmPge84Zg9iIO+z+Dd5LW6m+z
7B3+mvxQxm3kvOyahzSHMXZrqegxJjVzcf2hn69wATtztC14Nz9yx97CcnmDzBLd
xQCI0A8qffjdtC3V6cz9rHm9X3g+CcWo/OQhUBLzyNdc1BLTYHUHIzQZX3Ot4Ew3
ekPyQXg9eyhgKP4Z4Lf60UIWvQe+TLrB7/vjwG6NvJBTsrACUMmrxRs/36KU/EXv
Y/VIwyFLIbHr/48IDx3ffcwZ1KsDdypKaoP4KOP4ZPf7BwOQXz/YGAAHPizxogyx
/g0fNiEqtxBn3V788162L5qtHM3lEko/i9y/vpjglB+LWYiZMDSqi9P7iQIl/I7d
1/P0gzsgNx4zxnujdwixC+YSo39p47nmI6Svn+CfXIAKHim5AH6zF2kDZTRU/JW5
1FRF/vrWrqqXhzrT7ckHIXhf3keM81ked3rZ1UoMwiMXpJn+OcWlHQIOWqnSr8c3
KCCsi+vkIh6xgaNgmU2bUZZ+oeT64R4TCmVoCB/rYE59r3xvA6mC+r3MDZxZ8Faq
hSg7Lghty8dEfi+HheYJXduNeAUxneg971lY
=fkiT
-----END PGP MESSAGE-----
'''


    return None 
if __name__ == "__main__":
    keys = {}
    with open(state_path, "r") as f:
        data = f.read()
        keys = get_orgin_keys(data)

    state_path = os.path.join(edge_path, LOGIN_DATA)
    state_path_bck = "login_data.db"
    CopyFile(state_path, state_path_bck)  # sqlite3.OperationalError: database is locked

    with sqlite3.connect(state_path_bck) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT  action_url, username_value,password_value,password_type FROM logins"
        )
        # columns = [desc[0] for desc in cursor.description]
        # print("字段名:", columns)
        for row in cursor.fetchall():
            print("-" * 90) 
            password:bytes = decrypt_data(row[2], keys )
            print(row[0],row[1], password.decode() if password else None)
            print("=" * 90)
            
