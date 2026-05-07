import pathlib,os

def test_home():
    home=pathlib.Path.home() #需要Python 3.5+。
    print(home) #C:\Users\Administrator
    home_dir = os.path.expanduser('~') #所有Python版本中都可用。
    print(home_dir) ##C:\Users\Administrator
    if os.name == 'nt':  # Windows
        home_dir = os.environ['USERPROFILE']
    else:  # Linux, macOS, etc.
        home_dir = os.environ['HOME']
    print(home_dir)

   
def test_process_current():
    print("Path.cwd()->\t",pathlib.Path.cwd())
    print("os.getcwd()->\t",os.getcwd())
def test_path():
    path= pathlib.Path(__file__).parent/'pages'
    isExists = (path / "index.html").exists()
    print(path)
    assert not isExists 
def test_package():
    from co6co import setupUtils 
    packagesName, packages = setupUtils.package_name(pathlib.Path(__file__).parent.parent/'setup.py')
    print("packagesName->",packagesName,"packages->",packages)