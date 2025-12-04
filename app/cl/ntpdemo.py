import ntplib
from datetime import datetime, timezone


def get_ntp_time(ntp_server='ntp.aliyun.com'):
    """从指定NTP服务器获取时间"""
    client = ntplib.NTPClient()
    try:
        # 发送NTP请求
        response = client.request(ntp_server, timeout=5)

        # NTP时间戳转换为UTC时间
        utc_time = datetime.fromtimestamp(response.tx_time, timezone.utc)

        # 转换为本地时间
        local_time = utc_time.astimezone()

        return {
            'ntp_server': ntp_server,
            'utc_time': utc_time,
            'local_time': local_time,
            'offset': response.offset  # 本地时间与NTP时间的偏移量（秒）
        }
    except Exception as e:
        print(f"获取NTP时间失败: {e}")
        return None


if __name__ == "__main__":
    # 可以使用不同的NTP服务器，如'ntp.tencent.com'、'time.nist.gov'等
    # ntp_info = get_ntp_time('ntp.aliyun.com')
    # ntp_info = get_ntp_time('pool.ntp.org')
    import argparse
    parser = argparse.ArgumentParser(description="ntp service.")
    parser.add_argument('-n', '--ntp_server', type=str, default='ntp.aliyun.com', help='NTP server address')
    args = parser.parse_args()
    ntp_info = get_ntp_time(args.ntp_server)

    if ntp_info:
        print(f"NTP服务器: {ntp_info['ntp_server']}")
        print(f"UTC时间:  {ntp_info['utc_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"本地时间: {ntp_info['local_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"时间偏移: {ntp_info['offset']:.6f} 秒")
