
from co6co.utils import log
from sanic import Sanic
app = Sanic("MpApp")
data = globals()
log.warn(data)
mApp = data.get("data", None)
di = {}
if di:
    log.warn("NotisNUll", di)
else:
    log.warn("isNUll", di)
# log.info(__name__, app.name, id(app), id(mApp), '==>', type(mApp), mApp)
if __name__ == "__main__":
    globals()['mainApp'] = app
    app.run(host="0.0.0.0", port=8000, workers=1)
    log.warn("主进程是否推出？")
