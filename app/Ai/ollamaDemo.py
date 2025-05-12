# pip install ollama
import asyncio
from ollama import AsyncClient, Client
import ollama
import time
import os

host = os.getenv('OLLAMA_HOST')


def queryModule():
    """
    查询Ollama服务上可用的模型列表。
    该函数通过创建一个Ollama客户端实例，并调用其`list`方法来获取所有可用的模型。
    返回值:
        list: 一个包含所有可用模型信息的列表。
    """
    # 创建一个Ollama客户端实例，连接到指定的主机
    client = Client(host)
    # 调用客户端的`list`方法，获取所有可用的模型列表
    moduleList = client.list()
    # 返回模型列表
    return moduleList


def query(message):
    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': message,
        },
    ])
    print(response['message']['content'])


def localQuery(message: str):
    client = Client(host)
    response = client.chat(model='deepseek-r1:1.5b', messages=[
        {
            'role': 'user',
            'content': message,
        },
    ])
    print(response)


async def chat(message: str):
    message = {'role': 'user', 'content': message}
    response = await AsyncClient(host).chat(model='deepseek-r1:1.5b', messages=[message])
    print(response)
    return response['message']['content']


async def chat2(message: str):
    """
    stream=True 将函数修改为返回一个 Python 异步生成器
    """
    message = {'role': 'user', 'content': message}
    async for part in await AsyncClient(host).chat(model='deepseek-r1:1.5b', messages=[message], stream=True):
        print(part['message']['content'], end='', flush=True)

print(*queryModule())
localQuery('你是谁！')
asyncio.run(chat('你好！'))
asyncio.run(chat2('你是男是女?'))
time.sleep(1000*30)
