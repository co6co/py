from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.test_view  import TestsView,TestView

test_api = Blueprint("test_api" ) 
test_api.add_route(TestView.as_view(),"/test" )  
test_api.add_route(TestsView.as_view(),"/tests" )  