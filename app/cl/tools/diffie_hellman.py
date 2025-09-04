import math
import random

class DiffieHellman:
    def __init__(self, p=None, g=None):
        """
        初始化Diffie-Hellman密钥交换参数
        
        参数:
            p: 大素数
            g: 生成元
        """
        # 如果未提供参数，使用默认的预定义素数和生成元
        self.p = p if p else 23  # 默认使用小素数进行演示，可以替换为更大的素数
        self.g = g if g else 5   # 默认生成元
        self.private_key = None
        self.public_key = None
        self.shared_key = None
        
    def generate_private_key(self):
        """
        生成私钥，随机选择一个整数（1 < private_key < p-1）
        """
        # 确保私钥在有效范围内：1 < private_key < p-1
        self.private_key = random.randint(2, self.p - 2)
        return self.private_key
    
    def calculate_public_key(self):
        """
        计算公钥：public_key = g^private_key mod p
        
        返回:
            计算得到的公钥
        """
        if self.private_key is None:
            raise ValueError("必须先生成私钥")
        
        # 使用内置的幂取模运算，高效计算 (g^private_key) % p
        self.public_key = pow(self.g, self.private_key, self.p)
        return self.public_key
    
    def calculate_shared_key(self, other_public_key):
        """
        计算共享密钥：shared_key = other_public_key^private_key mod p
        
        参数:
            other_public_key: 对方的公钥
        
        返回:
            计算得到的共享密钥
        """
        if self.private_key is None:
            raise ValueError("必须先生成私钥")
        
        # 使用内置的幂取模运算，高效计算 (other_public_key^private_key) % p
        self.shared_key = pow(other_public_key, self.private_key, self.p)
        return self.shared_key

def is_prime(n, k=5):
    """
    简化版米勒-拉宾测试
    使用米勒-拉宾测试判断n是否为素数
    k为测试轮数，轮数越多，准确性越高
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    # 将n-1表示为 d*2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    
    # 进行k轮测试
    for _ in range(k):
        a = random.randint(2, n-2)
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        for _ in range(s-1):
            x = pow(x, 2, n)
            if x == n-1:
                break
        else:
            return False
    return True
def demonstrate_key_exchange():
    """
    演示Diffie-Hellman密钥交换过程
    """
    print("=== Diffie-Hellman密钥交换演示 ===")
    
    # 选择公共参数（实际应用中应使用更大的素数）
    p = 23  # 素数
    g = 5   # 生成元
    print(f"公共参数: p = {p}, g = {g}\n")
    
    # Alice的密钥生成过程
    alice = DiffieHellman(p, g)
    alice_private_key = alice.generate_private_key()
    alice_public_key = alice.calculate_public_key()
    print(f"Alice生成私钥: {alice_private_key}")
    print(f"Alice计算公钥: A = g^a mod p = {g}^{alice_private_key} mod {p} = {alice_public_key}")
    print(f"Alice将公钥 {alice_public_key} 发送给Bob\n")
    
    # Bob的密钥生成过程
    bob = DiffieHellman(p, g)
    bob_private_key = bob.generate_private_key()
    bob_public_key = bob.calculate_public_key()
    print(f"Bob生成私钥: {bob_private_key}")
    print(f"Bob计算公钥: B = g^b mod p = {g}^{bob_private_key} mod {p} = {bob_public_key}")
    print(f"Bob将公钥 {bob_public_key} 发送给Alice\n")
    
    # Alice计算共享密钥
    alice_shared_key = alice.calculate_shared_key(bob_public_key)
    print(f"Alice计算共享密钥: S = B^a mod p = {bob_public_key}^{alice_private_key} mod {p} = {alice_shared_key}")
    
    # Bob计算共享密钥
    bob_shared_key = bob.calculate_shared_key(alice_public_key)
    print(f"Bob计算共享密钥: S = A^b mod p = {alice_public_key}^{bob_private_key} mod {p} = {bob_shared_key}")
    
    # 验证双方计算的共享密钥是否相同
    print(f"\n验证密钥是否一致: {alice_shared_key == bob_shared_key}")
    print(f"双方共享密钥: {alice_shared_key}")


if __name__ == "__main__":
    # 运行演示
    # 注意：在实际应用中，应该使用更大的素数（如2048位或更大）来保证安全性
    # 这里为了演示简单使用了小素数
    demonstrate_key_exchange()

    big_prime = 2**2048   # 这只是一个示例，不一定是素数 
    # 检查其位数
    bit_length = big_prime.bit_length()
    print(f"位数: {bit_length}")  # 输出应该是1024
    for i in range(big_prime,big_prime+2**1024): 
        isPrime=is_prime(big_prime)
        if isPrime:
            print(f"找到素数: {big_prime}")
            break 

    # 可以直接进行算术运算
    result = pow(2, 512, big_prime)  # 计算2^512 mod big_prime
    print(f"2^512 mod {big_prime} = {result}")

