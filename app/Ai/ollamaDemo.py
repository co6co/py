# pip install ollama
import asyncio
from ollama import AsyncClient
from ollama import Client
import ollama
import time

host = "http://ps.co6co.top:65098"


def queryModule():
    moduleList = Client(host).list()
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
