import os
import hashlib
from co6co.utils import hash, log
from co6co.utils import find_files, convert_size,  convert_to_bytes, split_value_unit
import sqlite3 as db3
import argparse
from typing import Callable


def sqlExec(back: Callable[[db3.Connection, db3.Cursor], None]):
    '''
    CREATE TABLE "md5_data" (
    "id"  INTEGER PRIMARY KEY AUTOINCREMENT,
    "file_name"  VARCHAR(200),
    "md5_hash"  TEXT,
    "file_size"  INTEGER,
    CONSTRAINT "fileName_unique" UNIQUE ("file_name")
    );

    '''
    conn = db3.connect('md5_data.db3')
    cursor = conn.cursor()
    back(conn, cursor)
    conn.commit()
    conn.close()


def _computeMd5(current_dir: str, conn: db3.Connection, cursor: db3.Cursor):
    generFiles = find_files(current_dir)
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
                    _computeMd5(parent_dir, conn, cursor)
                elif os.path.isfile(item):
                    md5_hash = hash.file_md5(item)
                    fileSize = os.path.getsize(item)
                    log.info(f"计算：{item}的MD5,{md5_hash}")
                    cursor.execute(f"INSERT INTO md5_data (file_name, md5_hash,file_size) VALUES (?, ?,?)", (item, md5_hash, fileSize))
            except Exception as e:
                log.warn("error->", item,  e)


def compute_md5_values(current_dir):
    """计算指定文件夹中的所有文件的MD5值，并保存到数据库中."""
    # 创建数据库连接
    log.warn("current_dir:", current_dir)

    def _inner(conn: db3.Connection, cursor: db3.Cursor):
        _computeMd5(current_dir, conn, cursor)
    sqlExec(lambda conn, cursor: _inner(conn, cursor))


def show(md5: str, size: str, queryAll: bool = False):
    where = """ and md5_hash in (
        select md5_hash  from md5_data
        group by md5_hash
        HAVING count(*)>1
        )
        """ if queryAll else ''
    where = f"and md5_hash like '%{md5}%'" if md5 else ''
    where += f" and file_size > {convert_to_bytes(*split_value_unit(size))}" if size else ''

    sql = f'''
        select * from  md5_data
        where 1=1
        {where}
        order by md5_hash
    '''

    def _inner(_: db3.Connection, cursor: db3.Cursor):
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
    sqlExec(lambda conn, cursor: _inner(conn, cursor))
    # print(sql)


def delete(md5: str):
    sql = f"delete from  md5_data where  md5_hash='{md5}'"

    def _inner(_: db3.Connection, cursor: db3.Cursor):
        cursor = cursor.execute(sql)
        for c in cursor:
            print("删除结果：", c)
    sqlExec(lambda conn, cursor: _inner(conn, cursor))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MD5")
    parser.add_argument("-c", "--category", default='s', help="w:写入数据,d:删除指定md5值,s:显示重复数据[默认]")
    wGroup = parser.add_argument_group("写入数据")
    wGroup.add_argument("-d", "--dir",  type=str, help="目录路径")
    rGroup = parser.add_argument_group("md5")
    rGroup.add_argument("-m", "--md5", default=None, help="查询/删除 md5值对应的记录,默认为空")
    rGroup.add_argument("-s", "--size", type=str,  default=None, help="查询文件大小5MB,默认为空")
    rGroup.add_argument("-a", "--all", type=bool,  action=argparse.BooleanOptionalAction, default=False, help="查询所有")

    args = parser.parse_args()
    if args.category == 's':
        show(args.md5, args.size, args.all)
    elif args.category == 'w' and args.dir != None:
        compute_md5_values(args.dir)
    elif args.category == 'd' and args.md5 != None:
        delete(args.md5)
    else:
        parser.print_help()
