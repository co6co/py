from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
 
from view_model.biz.resource_view import Resources_View,Resource_View
from view_model.biz.poster_view import Poster_View,Image_View


resource_api = Blueprint("resource_API")

resource_api.add_route(Resources_View.as_view(),"/resource",name="resources")
resource_api.add_route(Resource_View.as_view(),"/resource/<uid:str>",name="resource") 

resource_api.add_route(Poster_View.as_view(),"/resource/poster/<uid:str>",name="viedo_poster")
resource_api.add_route(Image_View.as_view(),"/resource/poster/<uid:str>/<w:int>/<h:int>",name="image_poster")

