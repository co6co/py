import base64
import socket
# s=socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) #TCP
# s.connect(('192.168.2.108', 5050))
# s.send(b"hello world")

 
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.settimeout(10)  # 3 秒超时，防止卡死
# 绑定到 5051（你抓包里 PC 监听的端口）
s.bind(("192.168.1.99", 5051))
#s.sendto(b"discover", ("255.255.255.255", 5050))
#payload = b'\xa3\x01\x00\x01'+b'\x00'*12+b'\x02'+b'\x00'*15
payload = "owEAAQAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAA="
payload=  base64.b64decode(payload) if type(payload) == str else payload 

payload_hex = "a301000100000000000000000000000002000000000000000000000000000000"
payload = bytes.fromhex(payload_hex.replace(" ", ""))

s.sendto(payload, ("255.255.255.255", 5050)) 
#s.sendto(payload, ("255.255.255.255", 5050)) 
print("[*] 已发送 UDP 广播")
try:
    data, addr = s.recvfrom(4096)
    #print(data.hex())
    print(f"[+] 收到来自 {addr} 的回复：")
    print(data.decode(errors="replace"))
except socket.timeout:
    print("[-] 超时，未收到设备回复")
s.close()
