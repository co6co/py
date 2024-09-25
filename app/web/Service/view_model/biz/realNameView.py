
from sanic.response import text
from sanic import Request
from co6co_sanic_ext.utils import JSON_util
from co6co_sanic_ext.model.res.result import Result
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.sql import Select
from co6co_db_ext.db_utils import db_tools
from co6co_permissions.view_model.base_view import AuthMethodView
from model.pos.business import realNameAuthenPO
from view_model._filters.realName import Filter
from co6co_permissions.view_model.aop import exist, ObjectExistRoute


def calculate_check_digit(id_number):
    """
    输入身份证前17位
    计算出最后一位
    """
    if len(id_number) != 17:
        raise ValueError("ID number must be 17 digits long.")

    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_digits = "10X98765432"

    weighted_sum = sum(int(id_number[i]) * weights[i] for i in range(17))
    mod_result = weighted_sum % 11
    return check_digits[mod_result]


def check_digit(id_number):
    try:
        x = calculate_check_digit(id_number[:-1])
        if x == id_number[-1:]:
            return True
        return False
    except:
        return False


def mask_id_number(id_number):
    if len(id_number) != 18:
        return id_number

    masked_id = id_number[:6] + "********" + id_number[14:]
    return masked_id


class ExistView(AuthMethodView):
    routePath = ObjectExistRoute

    async def get(self, request: Request, code: str, pk: int = 0):
        result = await self.exist(request, realNameAuthenPO.identityNumber == code, realNameAuthenPO.id != pk)
        return exist(result, "证件号", code)


class Views(AuthMethodView):

    async def get(self, request: Request):
        """
        获取实名认证信息
        """
        userId = self.getUserId(request)
        select = (
            Select(realNameAuthenPO.id, realNameAuthenPO.name, realNameAuthenPO.category, realNameAuthenPO.phone,
                   realNameAuthenPO.identityNumber, realNameAuthenPO.authenState, realNameAuthenPO.createTime, realNameAuthenPO.imageBack, realNameAuthenPO.imageFront)
            .filter(realNameAuthenPO.userId == userId)
        )

        def maskIdentify(data):
            if data != None and "identityNumber" in data:
                data.update({"identityNumber": mask_id_number(data["identityNumber"])})
                mask_id_number

        return await self.get_one(request, select,  isPO=False, func=maskIdentify)

    async def post(self, request: Request):
        """
        树形 table数据
        tree 形状 table
        """
        param = Filter()

        return await self.query_page(request, param)

    async def put(self, request: Request):
        """
        增加
        """
        po = realNameAuthenPO()
        userId = self.getUserId(request)
        po.__dict__.update(request.json)
        po.userId = userId

        async def before(po: realNameAuthenPO, session: AsyncSession, request):
            try:
                if not check_digit(po.identityNumber):
                    return JSON_util.response(Result.fail(message=f"证件号'{po.identityNumber}'不正确"))

                exist = await db_tools.exist(session,  realNameAuthenPO.identityNumber.__eq__(po.identityNumber), column=realNameAuthenPO.id)
                if exist:
                    return JSON_util.response(Result.fail(message=f"证件号'{po.identityNumber}'已存在,请核对后再进行认证，如果用其他微信绑定请先解绑"))
            except:
                return JSON_util.response(Result.fail(message=f"证件号'{po.identityNumber}'不正确"))

        return await self.add(request, po, json2Po=False, userId=userId, beforeFun=before)


class View(AuthMethodView):
    routePath = "/<pk:int>"

    async def put(self, request: Request, pk: int):
        """
        编辑
        """
        po = realNameAuthenPO()
        po.__dict__.update(request.json)
        userId = self.getUserId(request)
        po.userId = userId

        async def before(oldPo: realNameAuthenPO, po: realNameAuthenPO, session: AsyncSession, request):
            if "*" in po.identityNumber:
                return
            if not check_digit(po.identityNumber):
                return JSON_util.response(Result.fail(message=f"证件号'{po.identityNumber}'不正确"))

            exist = await db_tools.exist(session, realNameAuthenPO.id != oldPo.id, realNameAuthenPO.identityNumber.__eq__(po.identityNumber), column=realNameAuthenPO.id)
            if exist:
                return JSON_util.response(Result.fail(message=f"证件号'{po.identityNumber}'已存在,请核对后再进行认证，如果用其他微信绑定请先解绑"))
        return await self.edit(request, pk, realNameAuthenPO, po=po, userId=userId, fun=before)

    async def delete(self, request: Request, pk: int):
        """
        删除
        """
        return await self.remove(request, pk, realNameAuthenPO)
