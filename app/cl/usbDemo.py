import usb.core
import usb.util
import traceback
import string,itertools
from co6co.utils import log
from co6co.utils import File
import time
# Change Fastboot idVendor and idProduct to your own
FB_IDVENDOR = 0x18d1
FB_IDPRODUCT = 0xd00d
# 查找所有 USB 设备
devices = usb.core.find(find_all=True)

# 遍历所有找到的设备
for device in devices:
    try:
        # 获取制造商字符串
        manufacturer = usb.util.get_string(device, device.iManufacturer)
        # 获取产品字符串
        product = usb.util.get_string(device, device.iProduct)
        print(f"Device: {manufacturer} - {product}")
    except usb.core.USBError as e:
        print(f"Could not fetch details for device: {device}, error: {e}")
        

def init_phone():
    dev = usb.core.find(idVendor=FB_IDVENDOR, idProduct=FB_IDPRODUCT)
    if not dev:
        print('Not found, exit')
        exit(-1)
    intf = dev[0][(0, 0)]
    usb.util.claim_interface(dev, intf)
    print('Init phone: finished')
    return dev, intf, intf[0], intf[1]


def clear_halt():
    phone.clear_halt(in_end)
    phone.clear_halt(out_end)
    print('Clear halt: finished')


def create_vendor():
    data_size = 1024
    data = b'\x00' * data_size
    return data, data_size


def send_bulk(cmd):
    assert phone.write(out_end, cmd) == len(cmd)
    ret = phone.read(in_end, 100)
    sret = ''.join([chr(x) for x in ret])
    print('Received from device:', sret)
    return sret


def flash_vendor():
    vendor, vendor_size = create_vendor()
    #print(vendor)
    #print(vendor_size)
    send_bulk('getvar:has-slot:vendor')
    send_bulk('getvar:max-download-size')
    send_bulk('getvar:is-logical:vendor')
    send_bulk('download:' + hex(vendor_size)[2:])
    send_bulk(vendor)
    send_bulk('flash:vendor')
    #print('Flash vendor: finished')
    #print(vendor)
    #print(vendor_size)


def unlock_vivo(dd ):
    try:
        flash_vendor()
        cmd = f'vivo_bsp unlock_vivo {dd}'
        s=send_bulk(cmd) 
        if 'Signature Fail' not in s:
            print("找到",s,dd)
            return True
        #print('Unlock: finished',s)
        return False
    finally:
         print("\n")


def lock_vivo():
    cmd = 'vivo_bsp lock_vivo'
    send_bulk(cmd) 
    print('Lock: finished')


print('Power off your phone, then press and hold Power + Volume Up to enter Fastboot mode')
input('If your phone is already in Fastboot, press Enter to continue:')
print('Available Operations:\n1: Unlock\n2: Lock')

phone, interface, in_end, out_end = init_phone()
op = input('Select an operation:')

try:
    if op == '1':
        clear_halt()
        characters = string.digits + string.ascii_lowercase
        # 定义组合长度
        num = input('input repeat Length:')
        # 生成所有可能的组合
        combinations = [''.join(c) for c in itertools.product(characters, repeat=int(num))]
        tmpFile="D:\\tmp\\a.txt"
        js:dict=File.File.readJsonFile(tmpFile) 
        old=js.get("index",0)  
        for index,a in enumerate(combinations):
            if index<=old: 
                print(f"从：{old}开始...")
                continue  
            
            if index % 1000 ==0:
                log.warn(f"************ {index} / {len(combinations)},{a}") 
                File.File.writeJsonFile(tmpFile,{"index":index,"value":a,"type":num}) 
                #time.sleep(0.5)
            if unlock_vivo(a):
                break
    elif op == '2':
        clear_halt()
        lock_vivo()
    else:
        flash_vendor()
        print('Invalid operation')
except:
    traceback.print_exc()
finally:
    usb.util.release_interface(phone, interface)