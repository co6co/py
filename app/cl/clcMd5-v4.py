import os
from functools import wraps
from co6co.utils import hash, log
from co6co.utils import find_files, convert_size,  convert_to_bytes, split_value_unit
import sqlite3 as db3
import argparse
from typing import Callable, Dict, List
import tempfile
from co6co.task import ThreadTask
from co6co.database.sqlite import ThreadSafeConnection


def exector(useDb3: bool = False):
    def decorator(f: Callable[[db3.Connection, db3.Cursor, List, Dict[str, any]], None]):
        """
        *args:int --> [Tuple[int, ...]
        **kwargs --> Dict[str, float]
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # print("arg:", args)
            # print("kwargs:", kwargs)
            # 禁用线程检查
            # 多个线程中同时操作同一个连接对象，可能会导致数据损坏或程序崩溃
            db = ThreadSafeConnection(os.path.join(kwargs.pop("folder", './'), 'md5_data.db3'), False)

            if useDb3:
                result = f(db, *args, **kwargs)
            else:
                with db as cursor:
                    result = f(db.conn, cursor, *args, **kwargs)
            db.conn.commit()
            db.conn.close()
            return result

        return decorated_function
    return decorator


@exector()
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


def filterFile(fileName: str):
    ignore = ['.vmdk']  # 该文件比较大一般不会重复
    # 可能返回 ''
    _, ext = os.path.splitext(fileName)
    if ext in ignore:
        return False
    return True


def contains_any_substring(path: str, subArr: list) -> bool:
    for substring in subArr:
        if substring in path:
            return True
    return False


def generate_task(current_dir: str, ignores: List[str] = None):
    """
    # 生成器函数，用于生成任务
    生成一个字符串
    """
    generFiles = find_files(current_dir, ['node_modules'],   filterFileFunction=filterFile)
    for r, _, flist in generFiles:
        ignores = ignores if ignores else []
        ignores.extend(['node_modules', '.pnpm-store', 'ProgramData', 'Program Files (x86)', 'Program Files', 'Windows', 'System Volume Information'])
        if contains_any_substring(r, ignores):
            log.warn("跳过目录", r)
            continue

        for file in flist:
            item = os.path.join(r, file)
            item = os.path.normpath(item)
            try:
                if os.path.isdir(item):
                    # 处理子目录
                    parent_dir = os.path.dirname(item)
                    generate_task(current_dir=parent_dir)
                elif os.path.isfile(item):
                    yield item
            except Exception as e:
                log.warn("error->", item,  e)


@exector(useDb3=True)
def computeMd5(db: ThreadSafeConnection, *, current_dir: str, maxThreads=4, ignore: str = None):
    log.warn("current_dir:", current_dir)

    def taskEnd(x):
        log.info(f"是否正常结束{x}")

    flag = 0

    def process_task(filePath: str):
        """
        处理任务的函数
        使用@exector,多线程下数据库被锁定
        """
        nonlocal flag
        flag += 1
        log.info(f"处理文件{filePath}")
        md5_hash = hash.file_md5(filePath)
        fileSize = os.path.getsize(filePath)
        log.info(f"{md5_hash}-->{filePath}:{fileSize}")
        try:
            with db as cursor:
                query = "select id,md5_hash from md5_data where file_name=?"
                cursor.execute(query, (filePath,))
                results = cursor.fetchall()
                # print("query results", results)
                if len(results) == 1 and results[0][1] != md5_hash:
                    # print("更新..")
                    cursor.execute("update md5_data set md5_hash=?,file_size=? where id=?", (md5_hash, fileSize, results[0][0]))
                elif len(results) == 0:
                    # print("插入..")
                    cursor.execute("insert into md5_data (file_name,md5_hash,file_size) values (?,?,?)", (filePath, md5_hash, fileSize))
                else:
                    log.info(f"{filePath} 存在， hash 未改变")
            if flag % 300 == 0:
                log.warn("提交", flag)
                db.conn.commit()
        except Exception as e:
            log.err("error->", filePath,  e)
        finally:
            pass
    ignores = ignore.split(',') if ignore else []
    task = ThreadTask(process_task, generate_task(current_dir, ignores),   taskEndBck=taskEnd)
    task.start(maxThreads)


@exector()
def show(conn: db3.Connection, cursor: db3.Cursor, *, md5: str, size: str, queryAll: bool = False, name: str = None):
    where = '' if queryAll else """ and md5_hash in (
        select md5_hash  from md5_data
        group by md5_hash
        HAVING count(*)>1
        )
    """
    where += f" and md5_hash like '%{md5}%'" if md5 else ''
    where += f" and file_size > {convert_to_bytes(*split_value_unit(size))}" if size else ''
    where += f" and file_name like '%{name}%'" if name else ''

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


@exector()
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
    wGroup.add_argument("-t", "--threads",  type=int, default=4, help="线程数")
    wGroup.add_argument("-i", "--ignore",  type=str, default=None, help="忽略的目录名称，多个用逗号分隔")

    rGroup = parser.add_argument_group("md5")
    rGroup.add_argument("-m", "--md5", default=None, help="查询/删除 md5值对应的记录,默认为空")
    rGroup.add_argument("-s", "--size", type=str,  default=None, help="查询文件大小5MB,默认为空")
    rGroup.add_argument("-n", "--name", type=str,  default=None, help="路径包含字符串")
    rGroup.add_argument("-a", "--all", type=bool,  action=argparse.BooleanOptionalAction, default=False, help="查询所有")

    args = parser.parse_args()

    if args.category == 's':
        show(md5=args.md5, size=args.size, queryAll=args.all, folder=args.folder, name=args.name)
    elif args.category == 'w' and args.dir != None:
        init(folder=args.folder)
        computeMd5(current_dir=args.dir, folder=args.folder, maxThreads=args.threads, ignore=args.ignore)
    elif args.category == 'd' and args.md5 != None:
        delete(md5=args.md5, folder=args.folder)
    else:
        parser.print_help()
