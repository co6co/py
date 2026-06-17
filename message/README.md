# 扩展 SQLAlchemy

SQLAlchemy 原始 使用起来总是有些不方便，该项目对其进行了部分封装为两个类：
DbOperations 和 DbPagedOperations 以方便业务使用；

ChunkedIteratorResult
✅ stream_results=True（语句 / 连接 / 引擎级）
✅ yield_per=N（语句 / ORM 级）
✅ session.stream(stmt)

1.  引擎级
engine = create_async_engine(
    "postgresql+asyncpg://...",
    execution_options={"stream_results": True}
)
2.  语句 / ORM 级
query = session.query(User).yield_per(500)
result = await session.execute(query)
3.  session.stream(stmt)
result = await session.stream(select(User))
# result 是 AsyncResult，内部包装 ChunkedIteratorResult
async for row in result:
    ...
4. stmt = select(User).execution_options(yield_per=200)
    result = await session.execute(stmt)
         
 1）「取单行」系列
.fetchone() → Row / None（底层游标取一行）
.first() → Row / None（结果集第一条，自动 LIMIT 1）
.one() → Row（必须唯一，无 / 多条抛异常）
.one_or_none() → Row / None（唯一或空，多条抛异常）
.scalar_one() → 单个值（第一行第一列，必须唯一）

2）「取多行」系列
.fetchall() → List [Row]（全量）
.fetchmany(size=N) → List [Row]（分批 N 条）
.all() → List [Row]（同 fetchall，ORM 常用别名）
.partitions(size=N) → 迭代器 [List [Row]]（分块迭代）

3）「剥 Row、取纯值」系列（重点）
.scalars() → ScalarResult（只保留第一列）
.scalars().fetchone() → 单个值 / 对象
.scalars().fetchall() → List [值 / 对象]
.scalars().one() / .one_or_none() → 同上
.scalar() → 单个值（第一行第一列，无则 None）

4）「字典化 / 列映射」系列
.mappings() → MappingResult（Row→dict）
.mappings().fetchone() → dict/None
.mappings().fetchall() → List[dict]
.keys() → 列名列表（如 ['id','name']）

5）「ORM / 异步特有」
.unique() → 去重（针对多列 / 关联查询）
.yield_per(N) → 流式缓冲 N 条（异步大结果必备）
.freeze() → 冻结结果（可重复遍历）


# (SQLALchemy Demo)[https://github.com/eastossifrage/sql_to_sqlalchemy/]

# 历史记录

```
0.0.1 初始版本
0.0.2 修复了部分 Bug
0.0.3 封装 JOIN 查询
0.0.6 2024-07-26
db_session.py  72 z
0.0.7 字符串格式
0.0.8 mapings(one)
0.0.9 优化
0.0.10
    Callable 中的类型 不能用 any需要用 Any
0.0.11
    BasePO 增加 assignment
0.0.12
    增加数据其他线程的数据库操作 -->session.BaseBll
0.0.13
    session.BaseBll 修复 connect 报错异常
0.0.14
    优化功能
0.0.15
    优化
0.0.16
	优化
0.0.17
    增加dbBll类，移除session中的DbSession、BaseBll类
0.1.0 2026-05-22
    增加 actuator，jwt_service
0.1.1 2026-05-23
    修复已知BUG
0.1.2 2026-06-01
    修复已知BUG
    cacheManage 缓存管理类 从其他项目中迁移过来做了关联调整
0.1.0601


```

- db_tools.execForPo

https://yifei.me/note/2652
https://www.osgeo.cn/sqlalchemy/
