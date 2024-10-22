from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
from co6co.utils.gps import dms_to_decimal


def parse_gps_info(gps_info):
    """解析GPS信息"""
    parsed_gps_info = {}
    for tag, value in gps_info.items():
        tag_name = GPSTAGS.get(tag, tag)
        parsed_gps_info[tag_name] = value
    return parsed_gps_info


def format_gps_coordinates(gps_info):
    """格式化GPS坐标"""
    lat_ref = gps_info.get('GPSLatitudeRef')
    lat: tuple = gps_info.get('GPSLatitude')
    lon_ref = gps_info.get('GPSLongitudeRef')
    lon = gps_info.get('GPSLongitude')
    latStr = dms_to_decimal(*lat, lat_ref)
    logStr = dms_to_decimal(*lon, lon_ref)
    return f"{float(latStr)}, {float(logStr)}"


def decode_user_comment(user_comment):
    """解码UserComment字段"""
    # 检查前两个字节是否为编码标识
    encoding = 'ascii'  # 默认编码
    if len(user_comment) >= 8:
        encoding_identifier = user_comment[:8].decode('ascii')
        if encoding_identifier == 'ASCII\x00\x00\x00':
            encoding = 'ascii'
        elif encoding_identifier == 'UNICODE\x00':
            encoding = 'utf-16'
        elif encoding_identifier == 'JIS\x00\x00\x00':
            encoding = 'shift_jis'
        elif encoding_identifier == 'UNICODE\x00':
            encoding = 'utf-16le'
        else:
            print(f"未知编码标识: {encoding_identifier}")

    # 解码UserComment字段
    try:
        decoded_comment = user_comment[8:].decode(encoding)
    except UnicodeDecodeError:
        print(f"解码错误，使用默认编码: {encoding}")
        decoded_comment = user_comment.decode('ascii', errors='ignore')

    return decoded_comment


def get_exif_data(image_path):
    """从给定的图片路径中提取EXIF数据"""
    try:
        image = Image.open(image_path)
        # 获取图片的EXIF数据
        exif_data = image._getexif()
        result = []
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'GPSInfo':
                    gps_info = parse_gps_info(value)
                    value = format_gps_coordinates(gps_info)
                elif tag_name == "UserComment":
                    value = decode_user_comment(value)
                result.append(f"{tag_name}: {value}")
            return result
        else:
            return ["没有找到EXIF信息"]
    except IOError:
        return ["无法打开或识别提供的图像文件"]


def main():
    data = get_exif_data("C:\\Users\\Administrator\\Desktop\\新建文件夹\\pmzi-04.jpg")
    return "\n".join(data)


print(main())
