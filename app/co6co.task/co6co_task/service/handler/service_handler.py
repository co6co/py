from co6co.task.eventDispatcher import EventHandler, Event, EventType
from typing import Optional
import time
from co6co_sanic_ext .sanics import IWorker
from sanic import Sanic
from co6co.utils import log
from .. import Scheduler
from ..CustomTask import ICustomTask
from co6co_web_db.services.db_service import BaseBll
from co6co_db_ext.db_utils import db_tools
from ..model.pos.tables import DynamicCodePO, SysTaskPO
from co6co_permissions.model.enum import dict_state
from sqlalchemy.sql import Select, Update
from typing import List, Tuple 

