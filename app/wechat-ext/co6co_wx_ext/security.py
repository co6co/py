import base64
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from .model import UserInfo



def decrypt_user_info(encrypted_data: str, iv: str, session_key: str)  :
    """
    wx.getUserInfo

    解密微信 encryptedData
    :param encrypted_data: 前端传来的 encryptedData
    :param iv: 前端传来的 iv
    :param session_key: 通过 code 换取的 session_key
    :return: 解密后的用户数据 dict
    """
    session_key = base64.b64decode(session_key)
    iv = base64.b64decode(iv)
    encrypted_data = base64.b64decode(encrypted_data)

    cipher = Cipher(
        algorithms.AES(session_key),
        modes.CBC(iv),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()

    decrypted = decryptor.update(encrypted_data) + decryptor.finalize()

    # PKCS7 去填充
    pad_len = decrypted[-1]
    decrypted = decrypted[:-pad_len]

    result = json.loads(decrypted.decode("utf-8"))
    user_info = UserInfo(**result)
    return user_info
