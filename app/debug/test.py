import requests,time

def checkService( ):
    try:  
        response=requests.get("http://127.0.0.1:8085/rest/debug",timeout=3)  
        if response.status_code==200:return True
        return False
    except Exception as e:
        print("error:",e)
        return False

while True:
    time.sleep(0.01)
    print("checked:", checkService())
