import hashlib,os

def md5(plain_text:str)->str:
    """
    获取MD5摘要
    """
    return str_md5(plain_text)

def file_hash(file_path: str, hash_method) -> str:
    if not os.path.isfile(file_path): raise Exception(message=f"{file_hash} not exist") 
    h = hash_method()
    with open(file_path, 'rb') as f:
        while b := f.read(8192): h.update(b)
    return h.hexdigest() 

def str_hash(content: str, hash_method, encoding: str = 'UTF-8') -> str:
    return hash_method(content.encode(encoding)).hexdigest()


def file_md5(file_path: str) -> str:
    return file_hash(file_path, hashlib.md5)


def file_sha256(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha256)


def file_sha512(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha512)


def file_sha384(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha384)


def file_sha1(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha1)


def file_sha224(file_path: str) -> str:
    return file_hash(file_path, hashlib.sha224)


def str_md5(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.md5, encoding)


def str_sha256(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha256, encoding)


def str_sha512(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha512, encoding)


def str_sha384(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha384, encoding)


def str_sha1(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha1, encoding)


def str_sha224(content: str, encoding: str = 'UTF-8') -> str:
    return str_hash(content, hashlib.sha224, encoding) 