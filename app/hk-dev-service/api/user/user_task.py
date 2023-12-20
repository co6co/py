from sanic import   Blueprint 
  
from view_model.task_view  import user_tasks_View, user_task_View
task_api = Blueprint("task_API") 
task_api.add_route(user_tasks_View.as_view(),"/biz/task",name="tasks") 
task_api.add_route(user_task_View.as_view(),"/biz/task/<pk:int>",name="task") 

 