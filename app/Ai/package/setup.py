from os import path
from setuptools import setup, find_packages


def get_version():
    """
    从包目录中的 versions.py 文件读取版本信息
    """
    current_dir = path.abspath(path.dirname(__file__))
    packages = find_packages()
    
    if not packages:
        raise RuntimeError("No packages found in current directory")
    
    package_name = packages[0]
    version_file = path.join(current_dir, package_name, 'versions.py')
    
    # 检查版本文件是否存在
    if not path.exists(version_file):
        raise RuntimeError(f"Version file not found: {version_file}")
    
    try:
        with open(version_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # 编译并执行代码
        compiled_code = compile(source_code, version_file, 'exec')
        scope = {}
        exec(compiled_code, scope)
        
        version = scope.get('__version__', None)
        if version:
            return version
            
        raise RuntimeError("Version variable __version__ not found in versions.py")
        
    except Exception as e:
        raise RuntimeError(f"Failed to read version from {version_file}: {e}")


def get_long_description():
    """
    读取 README.md 文件作为长描述
    """
    current_dir = path.abspath(path.dirname(__file__))
    readme_file = path.join(current_dir, 'README.md')
    
    if path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "No description available"


# 主要配置
if __name__ == "__main__":
    packages = find_packages()
    
    if not packages:
        raise RuntimeError("No packages found in the current directory")
    
    setup(
        name=packages[0],
        version=get_version(),
        description="web session 扩展",
        packages=packages,
        long_description=get_long_description(),
        long_description_content_type='text/markdown',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7", 
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9"
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[
            "co6co"
        ],
        author='co6co',
        author_email='co6co@qq.com',
        url="http://github.com/co6co",
        # 包含数据文件
        package_data={
            '': ['*.txt', '*.md'],
        }
    )
