# 未完成 
import os
from typing import Dict, List, Union
from multipart import parse_form_data,MultipartPart # pip install python-multipart,MultipartPart
from io import BytesIO
import json
from io import BytesIO
import os
from multipart import MultipartParser
import cgi  # 用于辅助函数
from typing import Tuple, Any



def safe_filename(filename):
    """生成安全的文件名"""
    if not filename:
        return "unknown_file.bin"
    
    # 去除路径信息，只保留文件名
    filename = os.path.basename(filename)
    # 替换可能危险的字符
    filename = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
    return filename

def parser_multipart_body(data:bytes, content_type:str) -> Tuple[Dict[str, tuple | Any], Dict[str, MultipartPart]]:
        """
        解析内容: multipart/form-data; boundary=------------------------XXXXX,
        的内容
        """
        env = {
            "REQUEST_METHOD": "POST",
            "CONTENT_LENGTH": len(data),
            "CONTENT_TYPE": content_type,
            "wsgi.input": BytesIO(data)
        }
      
        data, file = parse_form_data(env)
        print("in",data,file,len(data))
        data_result = {}
        # log.info(data.__dict__)
        for key in data.__dict__.get("dict"):
            value = data.__dict__.get("dict").get(key)
            if len(value) == 1:
                data_result.update({key: value[0]})
            else:
                data_result.update({key: value})
        print(data_result,file)

        return data_result, file
def parse_multipart_data(environ, content_length, charset='utf-8'):
    """
    完整的 multipart 数据解析函数
    """
    # 获取输入流和边界
    stream = environ.get("wsgi.input") or BytesIO()
    content_type = environ.get('CONTENT_TYPE', '')
    
    # 从 Content-Type 中提取 boundary
    if 'boundary=' in content_type:
        boundary = content_type.split('boundary=')[1]
        # 处理可能的引号
        boundary = boundary.strip('"')
    else:
        raise ValueError("Content-Type 中未找到 boundary 参数")
    
    result = {
        'fields': {},  # 文本字段
        'files': {}    # 文件字段
    }
    
    try:
        # 创建解析器
        parser = MultipartParser(stream, boundary, content_length, charset=charset) 
        for part in parser:
            field_name = part.field_name or f"field_{id(part)}"
            
            # 读取部分数据
            part_data = part.file.read()
            
            # 您代码中的核心判断逻辑
            if part.filename or not part.is_buffered():
                # 文件字段或需要流式处理的大数据字段
                filename = safe_filename(part.filename) if part.filename else f"file_{field_name}.bin"
                
                result['files'][field_name] = {
                    'filename': filename,
                    'data': part_data,
                    'content_type': part.content_type or 'application/octet-stream',
                    'size': len(part_data),
                    'field_name': field_name
                }
                
                print(f"检测到文件字段: {field_name}, 文件名: {filename}, 大小: {len(part_data)} 字节")
                
            else:
                # 普通文本字段
                text_content = part_data.decode(charset) if part_data else ""
                result['fields'][field_name] = text_content
                print(f"检测到文本字段: {field_name}, 内容: {text_content[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"解析过程中发生错误: {e}")
        return None

def parse_multipart_from_file(file_path: str, content_type: str) -> Dict:
    """
    从文件读取并解析multipart/form-data数据
    
    Args:
        file_path: 数据文件路径
        content_type: Content-Type头信息
        
    Returns:
        解析结果字典
    """
    with open(file_path, 'rb') as f:
        data = f.read()  
    return parser_multipart_body(data, content_type)
    #return parse_multipart_data(data, content_type)

def save_extracted_files(parsed_data: Dict, output_dir: str = './output'):
    """
    将解析出的文件保存到指定目录
    
    Args:
        parsed_data: 解析结果
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    for field_name, file_info in parsed_data['files'].items():
        filename = file_info['filename'] or f'{field_name}.bin'
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'wb') as f:
            if hasattr(file_info['data'], 'read'):
                # 如果数据是类文件对象
                file_info['data'].seek(0)
                f.write(file_info['data'].read())
            else:
                # 如果数据是bytes
                f.write(file_info['data'])
        
        print(f"文件已保存: {output_path}")


def demo1():
     # 示例1: 直接解析数据
     # 示例1: 直接解析数据
    # 假设你已经有原始的multipart数据和content-type
    example_data = """------WebKitFormBoundary7MA4YWxkTrZu0gW\r
Content-Disposition: form-data; name="text_field"\r
\r
这是一段文本\r
------WebKitFormBoundary7MA4YWxkTrZu0gW\r
Content-Disposition: form-data; name="file_field"; filename="example.txt"\r
Content-Type: text/plain\r
\r
文件内容在这里\r
------WebKitFormBoundary7MA4YWxkTrZu0gW--\r"""
    
    example_content_type = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    try:
        parsed_result = parse_multipart_form_data(example_data, example_content_type)
        
        print("=== 解析结果 ===")
        print("表单字段:")
        for field_name, value in parsed_result['fields'].items():
            print(f"  {field_name}: {value}")
        
        print("\n文件字段:")
        for field_name, file_info in parsed_result['files'].items():
            print(f"  {field_name}:")
            print(f"    文件名: {file_info['filename']}")
            print(f"    类型: {file_info['content_type']}")
            print(f"    大小: {len(file_info['data'])} 字节")
            # 如果是文本文件，可以显示内容预览
            if file_info['content_type'].startswith('text/'):
                try:
                    content_preview = file_info['data'][:100].decode('utf-8', errors='replace')
                    print(f"    内容预览: {content_preview}...")
                except:
                    print("    内容预览: [二进制数据]")
        
        # 保存提取的文件
        save_extracted_files(parsed_result)
        
    except Exception as e:
        print(f"解析错误: {e}")

def demo2():
    # 示例2: 从文件解析
    # 如果你的multipart数据保存在文件中，可以这样使用：
    filePath=input("输入multipart数据文件路径:")
    parsed_from_file = parse_multipart_from_file(filePath, 'multipart/form-data; boundary=--------------------------88def566461b568c')
    print("=== 解析结果 ===",parsed_from_file)
    save_extracted_files(parsed_from_file)

# 使用示例
if __name__ == "__main__":
    demo2()
    
    
    