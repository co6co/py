from co6co_web_db.view_model import BaseMethodView
from typing import Callable
import services.tasks.custom as custom


subclasses = custom.get_all_subclasses()
for subclass in subclasses:
    cl: custom.ICustomTask = subclass
    print(cl.name)
namelist = custom.get_list()
print(subclasses)
print(namelist)
