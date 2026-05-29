import asyncio
from services.ff_service import ffService
ff_service = ffService()

async def main():
    async for data in ff_service.read_rtsp_stream('rtsp://admin:lanbo12345@192.168.3.1/media/video1','1'):
        print(data)


asyncio.run(main())
   
"""sqlalchemy.engine.Engine SELECT sys_user_role.role_id
FROM sys_user_role, sys_user
WHERE sys_user_role.user_id = sys_user.id AND sys_user.id = %s UNION SELECT sys_user_group_role.role_id
FROM sys_user_group_role, sys_user
WHERE sys_user_group_role.user_group_id = sys_user.user_group_id AND sys_user.id = %s
2026-05-29 16:57:57,215 INFO sqlalchemy.engine.Engine [generated in 0.00059s] (1, 1)"""
