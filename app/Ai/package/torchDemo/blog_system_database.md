# 博客系统数据库设计

## 1. 数据库表结构

### 1.1 用户表（users）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 用户ID |
| username | VARCHAR(50) | NOT NULL, UNIQUE | 用户名 |
| password | VARCHAR(255) | NOT NULL | 密码（加密存储） |
| email | VARCHAR(100) | NOT NULL, UNIQUE | 邮箱 |
| nickname | VARCHAR(50) | NOT NULL | 昵称 |
| avatar | VARCHAR(255) | NULL | 头像URL |
| bio | TEXT | NULL | 个人简介 |
| status | TINYINT | NOT NULL DEFAULT 1 | 状态（1：正常，0：禁用） |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.2 角色表（roles）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 角色ID |
| name | VARCHAR(50) | NOT NULL, UNIQUE | 角色名称 |
| description | VARCHAR(255) | NULL | 角色描述 |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.3 用户角色关联表（user_roles）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| user_id | INT | NOT NULL, FOREIGN KEY REFERENCES users(id) ON DELETE CASCADE | 用户ID |
| role_id | INT | NOT NULL, FOREIGN KEY REFERENCES roles(id) ON DELETE CASCADE | 角色ID |
| PRIMARY KEY | (user_id, role_id) | | 联合主键 |

### 1.4 博客表（blogs）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 博客ID |
| title | VARCHAR(255) | NOT NULL | 博客标题 |
| content | LONGTEXT | NOT NULL | 博客内容 |
| summary | TEXT | NULL | 博客摘要 |
| author_id | INT | NOT NULL, FOREIGN KEY REFERENCES users(id) ON DELETE CASCADE | 作者ID |
| cover_image | VARCHAR(255) | NULL | 封面图片URL |
| status | TINYINT | NOT NULL DEFAULT 0 | 状态（0：草稿，1：已发布，2：已删除） |
| view_count | INT | NOT NULL DEFAULT 0 | 浏览量 |
| comment_count | INT | NOT NULL DEFAULT 0 | 评论数 |
| is_featured | TINYINT | NOT NULL DEFAULT 0 | 是否精选（0：否，1：是） |
| published_at | DATETIME | NULL | 发布时间 |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.5 分类表（categories）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 分类ID |
| name | VARCHAR(50) | NOT NULL, UNIQUE | 分类名称 |
| description | VARCHAR(255) | NULL | 分类描述 |
| parent_id | INT | NULL, FOREIGN KEY REFERENCES categories(id) ON DELETE SET NULL | 父分类ID |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.6 博客分类关联表（blog_categories）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| blog_id | INT | NOT NULL, FOREIGN KEY REFERENCES blogs(id) ON DELETE CASCADE | 博客ID |
| category_id | INT | NOT NULL, FOREIGN KEY REFERENCES categories(id) ON DELETE CASCADE | 分类ID |
| PRIMARY KEY | (blog_id, category_id) | | 联合主键 |

### 1.7 标签表（tags）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 标签ID |
| name | VARCHAR(50) | NOT NULL, UNIQUE | 标签名称 |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.8 博客标签关联表（blog_tags）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| blog_id | INT | NOT NULL, FOREIGN KEY REFERENCES blogs(id) ON DELETE CASCADE | 博客ID |
| tag_id | INT | NOT NULL, FOREIGN KEY REFERENCES tags(id) ON DELETE CASCADE | 标签ID |
| PRIMARY KEY | (blog_id, tag_id) | | 联合主键 |

### 1.9 评论表（comments）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 评论ID |
| blog_id | INT | NOT NULL, FOREIGN KEY REFERENCES blogs(id) ON DELETE CASCADE | 博客ID |
| user_id | INT | NOT NULL, FOREIGN KEY REFERENCES users(id) ON DELETE CASCADE | 用户ID |
| parent_id | INT | NULL, FOREIGN KEY REFERENCES comments(id) ON DELETE CASCADE | 父评论ID（用于回复） |
| content | TEXT | NOT NULL | 评论内容 |
| status | TINYINT | NOT NULL DEFAULT 1 | 状态（0：待审核，1：已发布，2：已删除） |
| like_count | INT | NOT NULL DEFAULT 0 | 点赞数 |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.10 评论点赞表（comment_likes）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| user_id | INT | NOT NULL, FOREIGN KEY REFERENCES users(id) ON DELETE CASCADE | 用户ID |
| comment_id | INT | NOT NULL, FOREIGN KEY REFERENCES comments(id) ON DELETE CASCADE | 评论ID |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| PRIMARY KEY | (user_id, comment_id) | | 联合主键 |

### 1.11 公众号配置表（wechat_configs）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 配置ID |
| app_id | VARCHAR(50) | NOT NULL, UNIQUE | 公众号AppID |
| app_secret | VARCHAR(255) | NOT NULL | 公众号AppSecret |
| token | VARCHAR(255) | NOT NULL | 公众号Token |
| aes_key | VARCHAR(255) | NULL | 公众号EncodingAESKey |
| status | TINYINT | NOT NULL DEFAULT 1 | 状态（0：禁用，1：启用） |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

### 1.12 公众号文章表（wechat_articles）

| 字段名 | 数据类型 | 约束 | 描述 |
|--------|----------|------|------|
| id | INT | PRIMARY KEY, AUTO_INCREMENT | 记录ID |
| blog_id | INT | NOT NULL, FOREIGN KEY REFERENCES blogs(id) ON DELETE CASCADE | 关联的博客ID |
| article_id | VARCHAR(50) | NULL | 公众号文章ID |
| title | VARCHAR(255) | NOT NULL | 文章标题 |
| status | TINYINT | NOT NULL DEFAULT 0 | 状态（0：待发布，1：已发布，2：发布失败） |
| error_msg | TEXT | NULL | 发布失败原因 |
| created_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP | 更新时间 |

## 2. 数据库关系图

```
+----------------+     +----------------+     +----------------+
|    users       |-----|   user_roles   |-----|     roles      |
+----------------+     +----------------+     +----------------+
       |                      |
       |                      |
       v                      v
+----------------+     +----------------+
|    blogs       |-----|  categories    |
+----------------+     +----------------+
       |                      |
       |                      |
       v                      v
+----------------+     +----------------+
|  comments      |-----|     tags       |
+----------------+     +----------------+
       |
       |
       v
+----------------+
| comment_likes  |
+----------------+

+----------------+     +----------------+
| wechat_configs |-----| wechat_articles |
+----------------+     +----------------+
                           |
                           |
                           v
                       +----------------+
                       |    blogs       |
                       +----------------+
```

## 3. 索引设计

### 3.1 用户表（users）
- 用户名索引：`username`
- 邮箱索引：`email`

### 3.2 博客表（blogs）
- 作者ID索引：`author_id`
- 状态索引：`status`
- 发布时间索引：`published_at`
- 浏览量索引：`view_count`

### 3.3 评论表（comments）
- 博客ID索引：`blog_id`
- 用户ID索引：`user_id`
- 父评论ID索引：`parent_id`

### 3.4 公众号文章表（wechat_articles）
- 博客ID索引：`blog_id`
- 文章ID索引：`article_id`
- 状态索引：`status`

## 4. 数据类型选择说明

1. **用户ID、博客ID等主键**：使用INT类型，自增，足够支持百万级数据量
2. **用户名、邮箱等字符串**：使用VARCHAR类型，根据实际需求设置合适长度
3. **密码**：使用VARCHAR(255)，因为加密后的密码长度较长
4. **博客内容**：使用LONGTEXT类型，可以存储大量文本内容
5. **状态字段**：使用TINYINT类型，只需要存储0、1、2等少数状态值
6. **时间字段**：使用DATETIME类型，支持精确到秒的时间存储

## 5. 约束设计说明

1. **主键约束**：确保每条记录的唯一性
2. **外键约束**：维护表之间的关联关系，确保数据完整性
3. **唯一约束**：确保用户名、邮箱等字段的唯一性
4. **非空约束**：确保必填字段不为空
5. **默认值约束**：为某些字段设置合理的默认值，如状态字段默认值为0或1

## 6. 分表分库策略（可选）

当系统数据量达到一定规模时，可以考虑以下分表分库策略：

1. **博客表分表**：按发布时间或作者ID进行分表
2. **评论表分表**：按博客ID或发布时间进行分表
3. **用户表分库**：按用户ID进行分库

## 7. 数据库备份与恢复策略

1. **定期全量备份**：每天凌晨进行一次全量备份
2. **增量备份**：每小时进行一次增量备份
3. **日志备份**：开启二进制日志，用于数据恢复
4. **异地备份**：将备份数据存储到异地服务器，防止数据丢失