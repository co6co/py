import os
from functools import wraps
from co6co.utils import hash, log
from co6co.utils import find_files, convert_size,  convert_to_bytes, split_value_unit
import sqlite3 as db3
import argparse
from typing import Callable, Dict
import tempfile


def exector(f: Callable[[db3.Connection, db3.Cursor, Dict[str, any]], None]):
    """
    *args:int --> [Tuple[int, ...]
    **kwargs --> Dict[str, float]
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # print("kwargs:", kwargs)
        conn = db3.connect(os.path.join(kwargs.pop("folder", './'), 'md5_data.db3'))
        cursor = conn.cursor()
        # print("kwargs:", kwargs)
        result = f(conn, cursor, **kwargs)
        conn.commit()
        conn.close()
        return result

    return decorated_function


@exector
def init(conn: db3.Connection, cursor: db3.Cursor):
    create_table_sql = '''
        CREATE TABLE IF NOT EXISTS "md5_data" (
        "id"  INTEGER PRIMARY KEY AUTOINCREMENT,
        "file_name"  VARCHAR(200),
        "md5_hash"  TEXT,
        "file_size"  INTEGER,
        CONSTRAINT "fileName_unique" UNIQUE ("file_name")
        ); 
        '''
    cursor.execute(create_table_sql)


@exector
def computeMd5(conn: db3.Connection, cursor: db3.Cursor, *, current_dir: str, ):
    log.warn("current_dir:", current_dir)
    generFiles = find_files(current_dir, ['node_modules'])
    flag = 0
    for r, _, flist in generFiles:
        # 遍历当前目录下的所有文件和子目录
        flag += 1
        if flag % 300 == 0:
            log.warn("提交")
            conn.commit()
        for item in flist:
            item = os.path.join(r, item)
            try:
                if os.path.isdir(item):
                    # 处理子目录
                    parent_dir = os.path.dirname(item)
                    computeMd5(conn, cursor, current_dir=parent_dir)
                elif os.path.isfile(item):
                    md5_hash = hash.file_md5(item)
                    fileSize = os.path.getsize(item)
                    log.info(f"{md5_hash}-->{item}")
                    cursor.execute(f"INSERT INTO md5_data (file_name, md5_hash,file_size) VALUES (?, ?,?)", (item, md5_hash, fileSize))
            except Exception as e:
                log.warn("error->", item,  e)


@exector
def show(conn: db3.Connection, cursor: db3.Cursor, *, md5: str, size: str, queryAll: bool = False):
    where = '' if queryAll else """ and md5_hash in (
        select md5_hash  from md5_data
        group by md5_hash
        HAVING count(*)>1
        )
    """
    where += f"and md5_hash like '%{md5}%'" if md5 else ''
    where += f" and file_size > {convert_to_bytes(*split_value_unit(size))}" if size else ''

    sql = f'''
        select * from  md5_data
        where 1=1
        {where}
        order by md5_hash
    '''

    cursor = cursor.execute(sql)
    field_names = [description[0] for description in cursor.description]
    # log.log("字段名", field_names)
    index = field_names.index('file_name')
    size_index = field_names.index('file_size')
    for c in cursor:
        temp_list = list(c)
        temp_list[index] = temp_list[index].replace("\\", "/").replace("//", "/")
        temp_list[size_index] = convert_size(temp_list[size_index])
        print(temp_list)

    # print(sql)


@exector
def delete(conn: db3.Connection, cursor: db3.Cursor, md5: str):
    sql = f"delete from  md5_data where  md5_hash='{md5}'"

    cursor = cursor.execute(sql)
    for c in cursor:
        print("删除结果：", c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MD5")
    parser.add_argument("-c", "--category", default='s', help="w:写入数据,d:删除指定md5值,s:显示重复数据[默认]")
    # dbFolder = tempfile.TemporaryDirectory().name
    dbFolder = tempfile.gettempdir()
    parser.add_argument("-f", "--folder", default=dbFolder, type=str, help=f"数据库所在文件夹default:{dbFolder}")
    wGroup = parser.add_argument_group("写入数据")
    wGroup.add_argument("-d", "--dir",  type=str, help="目录路径")
    rGroup = parser.add_argument_group("md5")
    rGroup.add_argument("-m", "--md5", default=None, help="查询/删除 md5值对应的记录,默认为空")
    rGroup.add_argument("-s", "--size", type=str,  default=None, help="查询文件大小5MB,默认为空")
    rGroup.add_argument("-a", "--all", type=bool,  action=argparse.BooleanOptionalAction, default=False, help="查询所有")

    args = parser.parse_args()

    if args.category == 's':
        show(md5=args.md5, size=args.size, queryAll=args.all, folder=args.folder)
    elif args.category == 'w' and args.dir != None:
        init(folder=args.folder)
        computeMd5(current_dir=args.dir, folder=args.folder)
    elif args.category == 'd' and args.md5 != None:
        delete(md5=args.md5, folder=args.folder)
    else:
        parser.print_help()
