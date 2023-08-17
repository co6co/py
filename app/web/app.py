import flask 
import argparse

app = flask.Flask(__name__)

@app.route('/')
def index():
    return "index"

@app.route('/test')
def test_api_request():
    return "06"

if __name__ == '__main__': 
    parser=argparse.ArgumentParser("webServer.")
    parser.add_argument("-p","--port",type=int,help="web service port", default=8080)
    parser.add_argument( "--debug",default=False, action=argparse.BooleanOptionalAction,help="debug ")
    args=parser.parse_args() 
    app.run('localhost', args.port, debug=args.debug)