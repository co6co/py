from setuptools import setup

setup(
    name='myPack', # 文件名
    version='1.0',
    packages=['myPack'],  # 替换为你的包名
    install_requires=[
        # 列出所有第三方依赖，例如
        'requests',
        'co6co',
        'numpy'
    ]
)
