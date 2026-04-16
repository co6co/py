import  os,sys
import argparse
 
def jk_sw_acl(args):
    mac_port_map = {}
    with open(args.source, 'r', encoding=args.encoding) as f:
        # 初始化字典来存储 MAC 地址和端口的映射
        mac_port_map = {}
        lines =f.readlines()
        for line in lines:
            # 使用 split() 默认按空白字符（空格、制表符等）分割，自动处理多余空格
            parts = line.split()
            # 确保每行至少有 3 列：MAC 地址、VLAN ID、端口
            if len(parts) >= 3:
                mac = parts[0]  # 第一列为 MAC 地址
                port = parts[2].split('/')[3]  # 第三列为端口
                if port in mac_port_map:
                    mac_port_map[port].append(mac)
                else:
                    mac_port_map[port] = [mac]
    
    with open(args.output, 'w',encoding=args.output_encoding) as f:
        for port,mac_list in mac_port_map.items(): 
            f.write(f"ipv4-mixed-access-list {port}\n")
            for i,mac in enumerate(mac_list):
                f.write(f"rule {i+1} permit {mac} 0000.0000.0000 any any\n")
            f.write("exit\n")
def main(args):
    if args.type == 1:
        jk_sw_acl(args)
    else:
        parser.error("Invalid type")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Text Tool")
    parser.add_argument('-s', "--source", type=str, help="Input file path")
    parser.add_argument('-e', "--encoding", type=str, default="utf-8", help="源文件编码")
    parser.add_argument('-o', "--output", type=str,   default="./dist/output/output.txt", help="Output file path")
    parser.add_argument('-t', "--type", type=int,   default="1", help="1:JK Switch ACL")
    parser.add_argument('-oe', "--output_encoding", type=str, default="utf-8", help="输出文件编码")
    args = parser.parse_args() 
    if not args.source:
        parser.print_help()
        sys.exit(1) 
    if not os.path.exists(args.source):
        parser.error("Input file not found")
    dir = os.path.dirname(args.output)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    main(args)