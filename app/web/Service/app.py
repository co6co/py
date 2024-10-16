from __future__ import annotations

from sanic import Sanic
from co6co.utils import log

from co6co_sanic_ext.utils.cors_utils import attach_cors
from co6co_sanic_ext import sanics
from co6co_web_db.services.db_service import injectDbSessionFactory
import argparse
from cacheout import Cache
import time


def init(app: Sanic, customConfig):
    """
    初始化
    """
    attach_cors(app)
    from api import api
    injectDbSessionFactory(app, app.config.db_settings)
    app.blueprint(api)
    cache = Cache(maxsize=256, ttl=30, timer=time.time, default=None)
    app.ctx.Cache = cache


def main():
    parser = argparse.ArgumentParser(description="audit service.")
    parser.add_argument("-c", "--config", default="app_config.json")
    args = parser.parse_args()
    sanics.startApp(args.config, init)


if __name__ == "__main__":
    main()
